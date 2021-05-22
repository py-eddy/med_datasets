CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��t�k       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�!�   max       Pz       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �\   max       <T��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @FH�\)     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @v��z�H     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @P�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��            7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <49X       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B0g�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��r   max       B0A�       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��   max       C���       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >>�   max       C��       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          V       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          W       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�!�   max       Px*       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���$tS�   max       ?�����       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <T��       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F<(�\     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @v��z�H     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�u@           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?wX�e+�   max       ?���Z��     �  ]               	         C      C   	   2                     #               '      V         
                      .   @   *   0   
                                 
                     
            U   "                     *N�vN#{Ng�O�PO��N��>NP��P��N ��Pb��N�qO�`�NűO'��P�OA(O6�O�xPJ�O�xO�nHN�XzO��O��aN�8_PzN;/O�\O!iNLjHP��O��aOp)O��nN.��P�xP= �Px*O���N߱�N�.�N �RNn�gNwaNI`.O2q&M��pN:	4N@ܔO$$iN���N�a`N���Ni�&N�LN�K�N�$�N�bEO�9KN��aM�!�P��O�Ne5N�xN��Np� N$�<Nt��O���<T��;��
;D����`B�o�o�o�T���e`B��o��o��o���㼬1��9X��9X��j��j�ě��ě��ě����ͼ��ͼ�����/��/��`B��`B��h�������o�o�+�+�\)�\)�\)�\)������w��w��w��w��w�#�
�#�
�0 Ž0 Ž0 Ž0 Ž@��P�`�Y��ixսq���q���u�y�#�}󶽃o�����O߽�O߽�����\#%&#

 

//14<AHUZ\]\XUJH<5//�����
�������!').5<@95)��������������������0<b����{n^J>57%��������������������KNgt����������_OJIKKtz����������������zt���������������������������������������������������������������������������������������������������������������������ht���������������hah�������� !��������������������������06COX\cig]K6*	��������������������`h�������������th\``th[OJIKO[ahit������t.56BN[_[[UNB5.......�/Uq���}{�znYXA���jmwz{zzzmldfjjjjjjjj��������������������5;HHTaggfaaUTRHD;005X[]gttttlg[XXXXXXXXX�� %��������}����������������~v}#2<E/)/<HKH8#
#0>Un{~yngU<0#	"&%"	x����������������ytx�����//)���������8t�����������g[VXPA8�*6BDFGJJB)������������������������U[gtv����tgd[WUUUUUUTUbnstrnjbZUTTTTTTTT��25BCNQQNB53,22222222��������������������)6BFLB@60)@BIOXUOOB>@@@@@@@@@@����������������������������������������tt������������xtsrot����������������������������������������{{�����������{trqt{{#/56/+#��������������������z��������������zzzzzyz{������������zyyyyqz�����������zwpqqqq����������������������������������,05<==<0-*,,,,,,,,,,����������������"#,/<><<;7/%#����������������������
�����)12+)$ABFNQ[__a[NKB@AAAAAAKN[gqg`[NFKKKKKKKKKKY[]`hlmptuthf[[VYYYY
!/<HPTVTH<#
�����������������������������������������������	�
��	���������������������׾ѾҾ׾�������׾׾׾׾׾׾׾׾׾������������������������������������������������������������������������������m�h�a�T�S�T�V�a�m�z�~�����z�m�m�m�m�m�m�U�O�J�I�Q�U�_�a�c�e�c�a�\�V�U�U�U�U�U�U�����������Ľ޽����A�Z�s�f�A��Ľ������������������������������������������s�l�c�k�y����������!�%�������������s�����������)�6�A�A�>�<�6�&�!��Y�M�@�'�����'�3�M�f�w�����������r�Y�U�M�I�<�7�5�<�I�U�b�d�n�s�s�n�b�U�U�U�U������������	���$�0�5�;�9�0�$������ۿҿ����"�5�A�N�\�c�b�P�(������àÜØÕÓÓÔÛàåìùû��������ùìà���������������������������������������̾�ھӾʾľþ������ʾ׾������������m�`�G�;�.�'�"�;�T�`�m�y�������������y�m�m�`�U�K�B�;�?�G�T�W�`�m�������������y�m�"�	����ݾվھ�����	��"�,�/�1�5�.�"�����������������������������������޻����n�o�f�`�_�l�x�����������������������`�\�\�T�H�/�"�����������$�:�;�G�H�a�`�V�J�I�C�=�?�I�V�W�b�j�o�m�b�V�V�V�V�V�V�ƿο������(�N�g�������	�,�%�������N�(��ƳƮƳƹ����������������ƳƳƳƳƳƳƳƳ�/�+�#���#�&�/�<�H�K�U�Z�_�U�T�H�<�/�/��
������(�5�A�B�N�O�N�F�A�5�(�������������"� ������������������������*�6�<�B�F�F�9�6�"����������սҽؽݽ�����!�#�"������¿²¡¦©²����������������������¿�����������{�s�k�i�o�s�������������������	����������	������	�	�	�	�	�	�	�	�e�Y�@�/���'�@�L�e�~���������������~�e�B�8�8�:�6�9�B�VĚĦħ����ļĦč�g�[�O�BĦėĚĦ��������#�<�L�Q�G�G�@�0����ĿĦ�s�r�i�b�_�W�W�l�x�����������»��������s�������������ɺֺ�����ֺֺɺ�������������������	���
�	����������-�,�&�)�-�:�F�S�F�?�:�0�-�-�-�-�-�-�-�-�ʾƾ¾ɾʾ̾׾������׾ʾʾʾʾʾʾ��������ƾʾ׾ݾ�پ׾ʾ���������������à×Øàììïù��ùîìàààààààà�s�p�h�t�������������������������������s�����������������������������������������z�u�m�h�a�^�a�m�w�z���������z�z�z�z�z�z����������������������������������������¦¦ ¢£¦²º¿������������������¿¦�������������������������������������������������
���#�%�#��
����������������������'�+�4�@�K�M�N�M�D�@�4�'����B�@�6�3�2�6�B�L�O�T�[�h�[�O�B�B�B�B�B�B��	���������� �����'�*�,�-�*���������ŹŵŴŹ��������������������������FcFWFVFJF=F<F1F*F1F=FJFVFVFcFcFgFcFcFcFc���������������������������������������������������ż������&�#�����ּʼ�������������������������������������������������������������Y�T�N�L�V�f����ϼѼμּܼռʼ������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�F E�E�E�EٽG�F�G�J�S�`�l�y�}�y�l�e�`�S�G�G�G�G�G�G��������ݽڽݽ����������������������ĽнҽнʽĽ��������������������z�y�n�d�a�Z�a�n�zÂÆÇÊÇ�z�z�z�z�z�zÓÊËÐÓÖàåæàÓÓÓÓÓÓÓÓÓÓ�������������������ùùϹϹйϹù�������EEEEE)E-E-E7ECEPE\EfEbEaEZEPEIE*EE - E ? 3 @ > � N W Y z J ^ 0 n b I b 0 V U S d N @ z Q ! 7 @ ( W ] D N @ E M 1 ? b [ L 5 H ; X r g E I D B g 7 3 u L Z ~ O ( ` ^ Y ! M [ ] X  -  9    L  _  �  �  x  8  C  Q  �  �  r  �  {  �  f  W  �  ~  �  �  �  �  	�  j  K  j  U  �  y  �  a  N  �  s    �    �  k  �  �  i  �    w  �  w    �  �  �  
  �  �  
  �  �  
  �  g  �    (  �  W  �  Y<49X;D��:�o���
��C��e`B���
���
��o����ě���������#�
�49X�'�P��w�m�h�D���H�9��w�8Q콃o�#�
��`B���]/��w�t���%�u�<j�P�`�t����w�ȴ9���㽧49X�49X�49X�8Q�,1�@��}�,1�,1�,1�aG��T���ixս<j�Y���o�q����o��O߽� Ž����%���Ƨ�\)������hs��{�� Ž\�C�Bh�B��B$t�B)�B��B�B rB&��B��B
R�B��B �0B��B��Bw�BO�B|�B�NB7rB+N6B0g�B��B|mB��B�B"rA��CBt�A��B	6@B�1B �0B kB&�uA���BϼB<oB
\B�B!%#B	rhB'��B�hB��B�BB�B��B�BCkB
��Bw9B�*B)mB��Bp�B
�+B��B��B-i@B�B%�B�B�B�B��B�B^iB�'B��B$B~/B��B$MmB SB�BtB1B&��B�yB
4�B��B аB�?B1�B@>B�KB}�B�vB��B*��B0A�BSvBx�BǗB�5Bc�A�w�BU>A�B	ADB�B @5B�5B&G�A��rBOkB�B	��B��B!;gB	§B'��BJvB�IB�xB?�B�;B8�B|�B
?MB�|B�"B)TB�B�dB4B@eB:B-��B�^B%�+B�qB��B�B�+B9BfBG�B�TBY�A�/3A���AT�3A��A���A�u�A�)�A+��A�f�A���A�c@�c,A�B	m-A��A�/�A���AT�Aj�FAkZlAZ�A�\�@�v�A�z7B�mA�ٍB�{AÐ�A���A�]_A��,A/�kA��#A�vxA�SW?�jPAܥA��	@�!@6�@AX��@{b�ASj�AQb�A�Y�A�7A��FA�	�A�<KA���A���A�֚@��A�21A���A�V�C���AIWXA�QB��@��o@���C�doA��A/Q�A&i�A�}A�֥=��C��pA�c�A���AU@UA�g�A�#A� �A�|ZA+�NA�~�A��Aր8@�l�A�cB	��A��EẢ�A�qAT�SAj$AkNAZrmA���@�m�A�;�B�A���B�AÇ�A�M�A��A���A1 A��WA���A�y�@z+Aݐ�A�kk@�-�@4"AZ`@t�AS��AR�
A̅*A�z�A�]yA�z4A���A��A�sbA�~@�A�LA�S�A���C��AI6A��BTe@�ob@���C�o�A�A1A%�eA��A�}h>>�C���               
         D      D   	   2                     #               '      V               !               .   @   *   0   
               	                  
                                 U   "                     *                        ?      5      #         +            '   !   !      !   %      W               %         '      +   -   3   !                                                            #         +                                                )      )               +               !                  7                        '      #   -   3                                                                                                N�vN#{Ng�O	�O��N��>N6��O�RbN ��P�(NE��N�p�N���O��P�OA(O�O�xO�]O�xOUO�N$�UO��OK�	N�8_Pt�N;/OĘO!iNLjHO�]�O�gOp)O��nN.��OP84TPx*O{��N߱�N�.�N �RNn�gNwaNI`.O2q&M��pN:	4N@ܔO$$iN���NU� N���Ni�&N��N�K�N�$�N�bEO���N��aM�!�OmL�O	��Ne5N�xN��Np� N$�<Nt��OE��  �  �  [  �  .  C  �  �  �  "  �    P    K  6  q    �  �  �  `  G  v    	0    �    �  $  �  U  �  R  ^  [    �    �  �  �  r  �  �  �  L  �  ~  `  K  �  �  �  /  l  u    V  �  	u  
Q  �  �  �  '  �  S  	<T��;��
;D���o�o�o�t����e`B�C������w��1�ě���9X��9X���ͼ�j�t��ě���`B�������t���/�<j��`B��h��h����w��w�o�o�+��P�t��\)�@��\)������w��w��w��w��w�#�
�#�
�0 Ž0 Ž8Q�0 Ž@��Y��Y��ixսq����%�u�y�#���ͽ�������O߽�O߽�������#%&#

 

6<DHUU[\[WUIH<700266�����
�������!').5<@95)��������������������#$0<IU`jqttqe^UH<0+#��������������������[gt���������ogXTPTV[�����������������������������������������������������������������������������������������������������������������������������������������ht���������������hah���������������������������������� *36CORZ\__UC6*��������������������dhu�������������th^dMOSX[ht{���}th[QOMM.56BN[_[[UNB5.......�
#HUn���rnaUF'�jmwz{zzzmldfjjjjjjjj��������������������5;HHTaggfaaUTRHD;005X[]gttttlg[XXXXXXXXX������������������������������#2<E/)/<HKH8#
#0>Un{~yngU<0#	"&%"	x|���������������zvx������-)��������8t�����������g[VXPA8��()6;=@?62)����������������������U[gtv����tgd[WUUUUUUTUbnstrnjbZUTTTTTTTT��25BCNQQNB53,22222222��������������������)6BFLB@60)@BIOXUOOB>@@@@@@@@@@����������������������������������������tt������������xtsrot����������������������������������������{{�����������{trqt{{#/56/+#��������������������z��������������zzzzzyz{������������zyyyyqz�����������zwpqqqq����������������������������������,05<==<0-*,,,,,,,,,,��������������������#,/<<<;<;6/$#����������������������
�����)12+)$ABFNQ[__a[NKB@AAAAAAKN[gqg`[NFKKKKKKKKKKY[]`hlmptuthf[[VYYYY
#*/<HOQKH</#	�����������������������������������������������	�
��	���������������������׾ѾҾ׾�������׾׾׾׾׾׾׾׾׾��������������������������
����������������������������������������������������m�h�a�T�S�T�V�a�m�z�~�����z�m�m�m�m�m�m�U�S�K�J�R�U�`�a�b�e�b�a�\�U�U�U�U�U�U�U�����������������Ľнݽ����-�4�)��ݽ���������������������������������������y��������������������������������)�!�&�'��)�4�6�<�9�6�6�)�)�)�)�)�)�)�)�Y�U�M�@�:�4�2�4�@�G�M�Y�f�r�r�|�v�r�f�Y�U�Q�I�=�I�U�b�n�p�q�n�b�U�U�U�U�U�U�U�U��������$�0�3�8�7�0�*�$��������ۿҿ����"�5�A�N�\�c�b�P�(������àÜØÕÓÓÔÛàåìùû��������ùìà���������������������������������������̾�ھӾʾľþ������ʾ׾������������y�m�`�T�K�?�;�T�]�m�y����������������y�m�`�U�K�B�;�?�G�T�W�`�m�������������y�m�	������ھ�������	��"�(�+�,�"��	�����������������������������������껅���x�o�p�g�a�`�l�x���������������������/�"��	�����"�/�6�H�R�T�V�V�U�H�C�;�/�V�J�I�C�=�?�I�V�W�b�j�o�m�b�V�V�V�V�V�V���
���0�A�Z�g�������������s�e�N�A�ƳƮƳƹ����������������ƳƳƳƳƳƳƳƳ�/�,�#���#�'�/�<�H�J�U�Z�^�U�S�H�<�/�/��
������(�5�A�B�N�O�N�F�A�5�(�������������"� ���������������������������*�1�:�=�8�/�*������������ܽݽ������������¿²¡¦©²����������������������¿�����������{�s�k�i�o�s�������������������	����������	������	�	�	�	�	�	�	�	�r�e�Y�3�)�(�@�L�e�~�����������������~�r�B�9�;�9�;�7�:�C�[ĚĦĳ����ĺĦč�[�O�BĦėĚĦ��������#�<�L�Q�G�G�@�0����ĿĦ�����|�q�j�l�m�|�������������������������������������ɺֺ�����ֺֺɺ�������������������	���
�	����������-�,�&�)�-�:�F�S�F�?�:�0�-�-�-�-�-�-�-�-�ʾƾ¾ɾʾ̾׾������׾ʾʾʾʾʾʾ��������ƾʾ׾ݾ�پ׾ʾ���������������à×Øàììïù��ùîìàààààààà�s�p�h�t�������������������������������s�����������������������������������������z�u�m�h�a�^�a�m�w�z���������z�z�z�z�z�z����������������������������������������¦¦ ¢£¦²º¿������������������¿¦�����������������������������������������
���������
���#�#�#��
�
�
�
�
�
�
�
������'�+�4�@�K�M�N�M�D�@�4�'����B�@�6�3�2�6�B�L�O�T�[�h�[�O�B�B�B�B�B�B���������������$�)�*�*�*�����������ŹŵŴŹ��������������������������FcFWFVFJF=F<F1F*F1F=FJFVFVFcFcFgFcFcFcFc�������������������������������������������������ʼּ߼�����"�������ּʼ��������������������������������������������������������������r�l�f�b�f�l�r���������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�F E�E�E�EٽG�F�G�J�S�`�l�y�}�y�l�e�`�S�G�G�G�G�G�G��������ݽڽݽ����������������������ĽнҽнʽĽ��������������������z�y�n�d�a�Z�a�n�zÂÆÇÊÇ�z�z�z�z�z�zÓÊËÐÓÖàåæàÓÓÓÓÓÓÓÓÓÓ�������������������ùùϹϹйϹù�������EEEE$E*E1E7E7ECEPE\E_E]E\E]EWEPECE*E - E ? / @ > � D W L w A Q ( n b . b + V I > a ; @ L Q   7 @ ' E ] D N ; B M " ? b [ L 5 H ; X r g E I @ B g ; 3 u L Q ~ O  \ ^ Y ! M [ ] ^  -  9    1  _  �  �  n  8  �  �    y  .  �  {  E  f  ,  �  �  -  �  �  �  :  j  <  j  U  �  }  �  a  N     W    �    �  k  �  �  i  �    w  �  w    h  �  �  �  �  �  
    �  
  �  c  �    (  �  W  �  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  [  X  V  T  R  O  M  K  H  F  E  F  G  H  H  I  J  K  K  L  �  �  �  �  �  �  �  �  y  f  O  2    �  �  �  c    �  e  .  ,  +  '  #          �  �  �  �  �  y  [  E  +  �  �  C  =  6  0  '          
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^    �  �  3  �  5  �  W  	  �  v  <  	  @  E  {  �  �  �  �  �  �  �  i  (  �  �  I  
  �  0  )   �  �  �  �  �  �  �  �  �  �  �  �  �  x  m  c  X  M  C  8  -  �  �  �      "      �  �  z  -  �  `  �  ^  �    C  �  �  �  �  �  �  �  �  �  �  �  �  �    �  G  l  h  c  \  U  �    A  n  �  �  �  �        �  �  �  X    �  I    �  <  7  3  3  ?  K  M  G  @  8  .  $    �  �  �  �    O    	              �  �  �  �  }  R    �  y  &  �  !  �  K  D  2       �  �  �  [  (  �  �  �  �  �  w  M    �  w  6    �  �  �  �  �  �  �  �  \  )  �  �  �  M    �  �  9  l  o  q  l  Z  D  +    �  �  �  �  w  Q  -    �  �  Y      w  k  ^  O  >  +    �  �  �  �  �  b  ;    �  }     �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  1  �  �  �  e  �  �  �  �  �  �  �  �  �  �  �  {  Z  3    �  �  E  �  j    �  �  �  �  �  �  �  �  �  �  z  ^  ?    �  �  �  w  3  �  �  �  	    0  M  \  `  b  ]  T  E  -    �  �  �  ~  H  
  E  G  E  >  1       �  �  �  �  �  �  H    �  �  ^     �  �  ,  O  g  r  v  p  [  8    �  �  u  7  �  �    }  �  Y    �  �  �  �  �  {  ]  =    �  �  �  ~  M    �  �  f  �  �  �    	   	0  	/  	%  		  �  �  ]  
  �  0  �  %  `  e    �    }  z  x  u  s  p  n  k  i  f  d  b  `  ^  \  Y  W  U  S  �  �  �  �  �  �  v  \  6  	  �  �  q  >    �  �  I  �       	  �  �  �  �  �  �  �  q  S  -    �  �  �  �  �  ~  l  �  �  �  �  �  �  �  �  �  �  �  e  D     �  �  �  {  O  #  �  �       "  "      �  �  �  �  I    �  S  �  _  �  k  �  �  �  �  �  �  �  �  �  �  u  U  1  �  �  G  �  =  |   �  U  B  *    �  �  �  �  '  *  &  !      �  �  �  �  �  �  �  �  �  �  �  u  d  P  =  ,         �  �  �  �  j     �  R  J  A  8  0  '      �  �  �  �  �  �  �  s  [  C  +    �  V  \  O  7    �  �  �  �  �  r  [    �  X  �  �    u  R  T  A  &  �  �  �  �  m  V  ?  !  �  �  W  �  &  d    �    �  �  �  m  5  �  �  �  �  �  �  |  I    �  [  �  U  �  y  �  �  �  �  �  �  �  �  �  ]    �  �  :  �  �  
  :  U        �  �  �  �  �  �  �  �  �  �  y  X  7    �  �  �  �  �  �  �  �  �  �  �  �  m  W  ?  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  e  L  2  �  �  �  �  �  �  �  �  �  �  �  y  g  Q  7      �  �  �  r  n  i  e  a  ]  Y  P  F  <  2  '       �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  ~  ^  =    �  �  �  w  M  #  �  �  |  h  L  -    �  �  �  �  p  Q  "  �  �    R  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  L  I  E  B  ?  <  8  5  2  /        �   �   �   �   �   m   S   9  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  u  m  i  g  d  \  P  @  ,      �  �  �  �  ]  -  �  �  `  Q  A  0      �  �  �  �  �  �  �  w  X  6    �  �  �    3  C  K  J  G  B  ;  5  .  %            �  �  x  1  �    {  v  q  m  h  b  [  T  M  F  ?  8  0  (           �  �               �  �  �  �  �  �  G  �  �  `     �  �  �  �  �  �  �  �  �  �  �  n  [  E  -    �  �  �  �  u  /  .  -  ,  %           �  �  �  �  �  �  �  �  f  H  )  l  Q  6    �  �  �  �  �  s  ^  I  3      �  �  u  .   �  u  f  X  K  >  .      �  �  �  �  �  r  M  %  �  �  �  �          �  �  �  �  �  �  f  ;    �  �  B  �  �  '  2  V  Q  M  H  C  @  =  9  9  <  ?  B  p  �    Q  f  m  t  {  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	E  	,  	%  	  	  	   	"  	I  	n  	t  	^  	'  �  ^  �  c  �  �  U  {  
D  
O  
>  
!  	�  	�  	�  	e  	7  	  �  �  &  �  V  �    V  �  �  �  �  �  x  l  _  S  F  8  (    	    *  B  [  X  P  G  >  �  �  �  �  �  �  �  �  �  �  �    h  R  =  (    �  �  �  �  �  �  �  �  �  �  �  �  �  }  q  e  Y  M  A  5  *      '  $               	    �  �  �  �  �  �  �  �  �    �  �  �  �  �  
    #  0  <  G  P  Y  b  l  y  �  �  �  �  S  E  7  *  1  <  C  :  (    �  �  �  �  z  L    �  w  '  	  �  �  	  	  �  �  l  +    "  <  I  �  �    �    ]  