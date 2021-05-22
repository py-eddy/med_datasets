CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�p��
=q       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�md   max       P���       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(��   max       @FǮz�H     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v\(�     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @N�           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @��`           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �V   max       =t�       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�C+   max       B5       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��,   max       B4�w       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >yV�   max       C���       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >C��   max       C��       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          t       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�md   max       Pe�       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����+   max       ?�:�~���       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\   max       =��       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @FǮz�H     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v~�\(��     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @N�           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�            V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D   max         D       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?vOv_خ   max       ?�64�K     P  X�                           
               J                  1         Y      s      	      "            	      	   	            R   +       	   )                           	      '      )      
                  M�mdN���N�.XOW5>O�h�N�=Nı�N3mwNM�hO�U~N�$�O\8�O�x�P\�N2�O�
>O��XOcm�NtEgP5�P	��O���PloO�P���N���N4�EN)y�O��<O��CN��O�vTO�6zO��SO.NꃜN�`�N���N��XP:l%P�O���NՒSOp;�N��jN��{O��8N���OTVqN��O"�kN�N�%�O��/O�#?NpO>�\O�@�OΦOl�N̄EN"\�O�+N�X�N�.=��<��
<#�
;�o;D��;o�o�D����o���
���
�o�#�
�#�
�#�
�#�
�#�
�T���e`B�u���
��1��1��9X�ě��ě�������/��/��/��h��h�o�o�o�C��C��C��\)�\)�\)�t��t���w��w�#�
�8Q�<j�@��D���H�9�m�h�u��������7L������-���-���w�������T��1����������������������������������������������������������������������������������������������������������
#%020# 
fhmtu�����������thff����

����������#%"������������������������ ���������#',,/<AFKD<4/#�������������������)5BNbklkgb[N5)#*/01/#058BFO[�����tOB6)'+6@[ty}{o_[OB?6)'('����������������������������������������6BU]beLB)����  
��������2AN[gtvzxuoiig[NLH;2#0<Pmx|}snbI<-#���������������������gp|�������������zj\g���������������U[htwtmh[RUUUUUUUUUUlmprxz�����zzmllllll*/<=BHV[^^`UH#)66CJV\^[OB6)"����������$)1BN[gt}����tg[5+%$�������
 �����
#(IUZbbTPI93.#

abn{����������{nbdca����������������������� ���������������������������hmoz����������zumjhh����5?5���������y����������������|wy��������������������eht�����~zthh_]]eeee������ �����������������������������������������������������
#;BB</#
���!##0<BIIIIC<0+#"!!!!U[`gt��������tg[VRSUBBFOO[]hiklh[OEBBBBB�ztg][VSTVX[gjprt{�������������������������������������}{����������������~�����5BNZUQNLHB5)
�������������������������
#"
������������ ����������� !���� #.0<DHIIIIHF<3/*#wz{�����������zytoww��������������������������������
 "#+#
�-/0<BHOUZ^[UH<</----¦¦¦�!� �!�%�-�5�4�:�D�F�N�S�X�S�F�E�:�-�!�!�H�D�<�7�7�<�H�U�Z�a�n�w�z��z�n�a�^�U�H��������������
��#�/�<�H�R�H�<�/�#��ܻû������������ûлܻ����(������ܼ����������������������������������������������������������������¿¸²©²¹¾¿��������¿¿¿¿¿¿¿¿��غֺͺֺ������������������`�K�B�;�9�;�@�G�T�m�u���������������y�`���������������Ŀѿٿտݿݿ�ݿӿѿĿ����H�<�2�/�#����#�/�U�a�g�n�u��z�n�U�H�g�Z�N�F�@�D�N�Q�Z�g�s�w�������������s�g�Z�S�A�*�$�!�$�5�N�g�s�������������s�g�Z�)�)����%�)�5�7�<�8�5�)�)�)�)�)�)�)�)�������m�`�T�J�I�m�y�����������������¿������������žʾ��	�.�@�:�=�.�#����ʾ��׾վʾɾǾƾʾо׾����	��������׼Y�Q�Q�Y�f�r�}�z�r�f�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�F�;�=�:�=�F�S�l�����������������l�_�S�F���v�p�u�������ʾ׾������	��ʾ��������������������)�5�N�[�a�c�J�B�5�������v�s�v�����н��A�M�X�R�@���ҽ��������
���(�5�;�=�<�5�5�(������������_�O�F�:�:�F�_�l�����û���� ���ܻ����ֹܹϹù������¹ùϹعܹ��������ù������ùƹϹֹܹϹùùùùùùùùù��(�(�����������(�)�)�(�(�(�(�(�(�������~�t�������������������������������g�a�^�g�s���������������������������s�g�ʾ��¾ʾ׾�����	��	�����׾ʾʾʾʾ����վϾҾپݾ����	������	����y�b�T�M�G�;�2�>�G�T�`�m�y�����|�������y�����������������������������������������Z�Y�G�9�6�A�N�P�Z�g�i�s�����������s�g�Z�����������������
��#�'�(�#���
� ���������������������������������������������� �%�*�3�*�!������������������������$�%�%�$��������+�[ĎĹĿ����������ĿĚā�O�B�)��	�����/�T�a�m�y���������a�H�;�"�	�[�R�P�Z�hƁƧ��������������ƳƧƚƎ�h�[�	���	�	��"�+�/�;�C�;�;�/�"��	�	�	�	�������������*�6�C�F�L�K�C�6�*� �����������������������ʾ;о;ʾþ�������ìáàÞàâìù��������þùìììììì������
���)�6�B�U�V�V�Z�S�O�B�6�,��������������'�2�/�'���������	������"�/�;�H�H�L�K�A�;�/�"�ààÓÇÇÄÇÒÓàìòõðìáààààƎƚƤƢƚƘƎƁ�u�h�\�U�M�O�\�h�k�uƁƎ�e�[�Y�V�Y�e�l�r�w�{�r�g�e�e�e�e�e�e�e�e�����������ûĻлܻ������ܻлû���������������$�/�8�=�I�P�X�V�J�=�0�$�����U�I�C�G�H�F�H�a�zÊÓàçß×É�x�n�a�U��������������	�����������������ED�D�D�D�D�D�D�D�EEE"E*E/E-E*E"EEE�E�6�2�7�?�G�S�`�y�������������z�l�`�S�E�����������Ľн�����������ݽнĽ�FFE�E�FFFF$F1F=FJFLFJFHF=F;F1F&FF�ϹϹϹԹܹ߹��������������ܹϹϼ����������ʼּ���ּּʼ���������������w�r�f�Y�W�P�Q�Y�f�r�w����������������߿޿������������������������� �����(�.�/�(�(����� C P @ a L t L \ ` ? T Q 0 ! 4 e :  "   H R [ ( \ { L f B 7 N G M ? 0 4 - D + y D b ! 2 4 2 I   & K M K h 7 a < ) h P O v c B 6    +  �  
  �  �  �  �  �  Z  �    �  �  I  ?  =  k  �  {  j  r  a  I  M  �  �  Z  ~  �  !  8  �  Q  x  o    �  �    y  �  '  �  �    �  :  �  �  �  o  5    �  �  M  �  .  �  [    }  `  �  �=t�<u��o�49X�ě�%   �o��o�e`B�+�#�
��h��`B����e`B�ě��'�h���ͽ�o�49X�t���G��,1�V���\)�+�y�#�D���#�
�H�9�'H�9�',1�L�ͽ��T������-��7L�8Q콡���Y��Y�����e`B�ixսq����o�����C���-��񪽏\)��xս�"ѽ� Ž�/��^5��1����������BA|B,�nB!v�B�B k-B$�	B�wBhzB�YB+6_B-�;B$B�mB��B��B�|B�bB6�B!B�B#g8B�B&_OB�cB�ZB�FB�TA�C+B��B��Ba�B	�B�B%��B(�B�XB�yB#B B��B6,B(B �Bg�B5BΖB[�B%��B	�iBVB	K�B��B)��B`B� B"�By�Bi�B�BwcB��B,�:BRqB"B1YBAB,�B!�xBǶB L�B$�B�9B@�B�B+	�B-`�BB^�B��B�uBEB�!B>BB ��B�dB#�dB��B';�B{]B?�B>�BF%A��,B��B��B7FB	�BͿB&?B)+B�6B��B
�B 	B��B7
BB?�B�zB4�wB�B�B&&OB	y�B?�B	CVB�B)�XB�]B�B"çB�B��BANB@B�B,�GB�<B�1BC�A�J�@}gfA���A�.x@�1R@�n�A1��A�@Ho�AjAy�-A�.*A���A��A��Ap�5AV��AV1T@��@�>�AO��A�:�A+��A��H@���>�n>yV�A�	A��bA�ʑAU�!AX� Aj_�A��FA�
�A�c�A��[A��B��A�`�A�M�BQ�A�]�A���AM?xA��A��U@��A�	TA�l�B��?��@���B
�A�7�@Z�=C�\*A��A+j6C���?[�@�q%@ᢾA��|A�>wA�|@|C�AƈLA���@��@�A2�qA�x�@D0TAi�Az2�A�aNA��A�z�A��An��AW^AV�@�Ф@� #AJlA�\XA.	�A�$�@���>���>C��A��FA�v�A��=AT=CAY��AiOA���A�S!A�zA���A��B�AA�][A��nB�IA��.A�'�AL�EA���A�~�@���A��nA�w�B)?��H@��tB	/SA�i�@[B?C�XGA�A+��C��?.�=@�:�@�SA�k�A�{�                                          J                   2         Z      t      
      #            
      
   	            S   +       
   )                           	      '      *      
                                 '               !            #      '   )         #   )      9      9            !               #                  5   )                                       !                                                #                                 '   '            )      1      +                                                #                                       !                                 M�mdN.�NѮrO"$sO�԰N�=Nı�N3mwN2k�Oc�N�t:N��XO]�'O�k�N2�O�
>O��nO,NtEgO��P	��O���Pe�O�P
EN���N4�EN)y�O��O>��N��O%x O�6zO��O.NꃜN�`�N���N�okO�ƹO���O�s�NՒSO9;N��jN��{O��8N��bOTVqN��O"�kN�N�%�O��/Or�NpO>�\O�@�OΦN���N̄EN"\�O�+N��dN�U  �  �  �  W    �  �  �  �  �  �  �  $  	  �  �  �  ;  b  P  �  q  	  G  Y  \  �  W  =  �  t  �  �  �  �  ]  w  �  i    �  �    �  H  �      
  i  �    �  5  �  N  
�  �    �  �  �    &  �=��<���<t�:�o;o;o�o�D�����
�e`B�ě���t��D���o�#�
�#�
�D����o�e`B�0 ż��
��1�@���9X�}�ě�������/�+�o��h�t��o��w�o�C��C��C��t���7L�#�
�#�
�t��8Q��w�#�
�8Q�@��@��D���H�9�m�h�u��������7L������-���-�����������T��1�\�\����������������������������������������������������������������������������������������������������
#%020# 
fhmtu�����������thff����

����������"$!����������������������������������#%/6<=><6/#�������������������� &05BN[^bdb[NB5/##*/01/#058BFO[�����tOB6))/6<[tw|ynha[OB8*(*)����������������������������������������#)/6BKOQSROIB6,)"##����  
��������2AN[gtvzxuoiig[NLH;2#0<JgpsshbUI<5+���������������������������������zqnos���������������U[htwtmh[RUUUUUUUUUUlmprxz�����zzmllllll#(/<HMSUYUMH</#)6=CNVWOB6/) ����������JN[gqt}����tg`[NIDBJ�������
 �����#*0<ISUUIH<0#abn{����������{nbdca����������������������� ���������������������������impz����������zxmkii�����	������}����������������{}��������������������eht�����~zthh_]]eeee������������������������������������������������������������
#;BB</#
���##%0<@GHB<0-########U[`gt��������tg[VRSUBBFOO[]hiklh[OEBBBBB�ztg][VSTVX[gjprt{�������������������������������������}{����������������~����� BLNONLHB5) 
�������������������������
#"
������������ ����������� !���� #/<CGHHIHG<4/+#"wz{�����������zytoww��������������������������������
 
�3<EHKUZ]ZUH=<1333333¦¦¦�-�)�'�-�:�F�S�T�S�F�<�:�-�-�-�-�-�-�-�-�H�D�<�9�;�<�H�U�[�a�n�v�z�}�z�n�a�\�U�H�#������������
��#�/�:�<�A�H�H�<�/�#��ܻû����������ûлܻ�����&�����鼱���������������������������������������������������������������¿¸²©²¹¾¿��������¿¿¿¿¿¿¿¿��ںֺκֺ������������������`�T�J�C�A�D�G�T�`�m�v�y�����������y�m�`�Ŀ������������ÿĿ̿ѿѿۿݿ�ݿѿ̿Ŀ��H�=�<�6�<�D�H�U�X�a�h�j�a�U�H�H�H�H�H�H�g�Z�Q�N�H�B�G�N�X�Z�g�s�����������}�s�g�g�Z�N�A�5�0�-�0�;�A�N�g�s�����������s�g�)�)����%�)�5�7�<�8�5�)�)�)�)�)�)�)�)�������m�`�T�J�I�m�y�����������������¿������������Ǿʾ��	��"�1�6�0�"����ʾ���߾׾ξ˾ʾʾ׾پ�����	���������Y�Q�Q�Y�f�r�}�z�r�f�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�_�X�S�O�N�S�W�_�l�x�~�����������x�l�_�_���v�p�u�������ʾ׾������	��ʾ��������������������)�5�N�[�a�c�J�B�5���������������Ľݽ���4�A�F�@�(��ݽ����������
���(�5�;�=�<�5�5�(����������l�^�_�l���������û������ܻлû����ֹܹϹù������¹ùϹعܹ��������ù������ùƹϹֹܹϹùùùùùùùùù��(�(�����������(�)�)�(�(�(�(�(�(�����������������������������������������s�q�k�l�s�����������������������������s�ʾ��¾ʾ׾�����	��	�����׾ʾʾʾʾ�����������	�	������	�����y�b�T�M�G�;�2�>�G�T�`�m�y�����|�������y�����������������������������������������Z�Y�G�9�6�A�N�P�Z�g�i�s�����������s�g�Z�����������������
��#�'�(�#���
� ���������������������������������������������� �%�*�3�*�!������������������������$�$�$�$�����h�[�O�T�[�g�tāĚĦĳĿ����ĽĳĪĚā�h������"�1�T�a�m�t�����q�a�H�;�"��u�i�]�[�h�mƁƧ������������ƳƧƚƎƁ�u�	���	�	��"�+�/�;�C�;�;�/�"��	�	�	�	�� ��������*�6�B�C�I�G�C�7�*������������������������ʾ;о;ʾþ�������ìáàÞàâìù��������þùìììììì������
���)�6�B�U�V�V�Z�S�O�B�6�,������������'�0�,�'�����������	������"�/�;�H�H�L�K�A�;�/�"�ààÓÇÇÄÇÒÓàìòõðìáààààƎƚƤƢƚƘƎƁ�u�h�\�U�M�O�\�h�k�uƁƎ�e�[�Y�V�Y�e�l�r�w�{�r�g�e�e�e�e�e�e�e�e�����������ûĻлܻ������ܻлû���������������$�/�8�=�I�P�X�V�J�=�0�$�����U�O�H�H�O�U�]�a�n�zÇÔ×ÑÇÃ�r�n�a�U��������������	�����������������ED�D�D�D�D�D�D�D�EEE"E*E/E-E*E"EEE�E�6�2�7�?�G�S�`�y�������������z�l�`�S�E�����������Ľн�����������ݽнĽ�FFE�FFFF$F$F1F7F=FGF=F:F1F%F$FFF�ϹϹϹԹܹ߹��������������ܹϹϼ����������ʼּ���ּּʼ���������������w�r�f�Y�W�P�Q�Y�f�r�w����������������������������������������������(�-�.�(�&������� C @ A S I t L \ V ; V $ /  4 e 6  "  H R f ( e { L f + 6 N 7 M @ 0 4 - D ' ^ D b ! + 4 2 I   & K M K h - a < ) h C O v c . 3    +  S  �  �  b  �  �  �  I  �  �  �  �  2  ?  =  #  m  {  /  r  a  ]  M    �  Z  ~  :  �  8  k  Q  S  o    �  �    z    �  �  �    �  :  �  �  �  o  5    �  �  M  �  .  �      }  `  �  �  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  �  �  ~  {  w  s  o  l  h  d  [  N  A  3  &      �  �  �  k  r  y  �  �    y  s  l  b  Y  O  @  /       �   �   �   �  �  �  �  �  �  �  �  �  �  �  q  G    �  �  r  7    �  �  H  I  Q  W  W  S  I  <  )    �  �  �  Z    �  m  :  )            �  �  �    �  �  �  �  Q    �  r  
  �  Y  _  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    h  M  2  !              '  1  <  �  �  �  �  �  �  �  �  ~  x  q  j  b  V  ;      �  �  �  �  �  �  �  �  �  �  y  c  N  2    �  �  X  �  [  �  v  	  Z  �  �  �  �  �  �  �  �  �  �  �  }  [  0  �  �  c  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  t  l  Y  C  -    �  �      .  M  l  �  �  �  �  �  o  T  7    �  �  z  k  #  $  $         �  �  �  �  v  M  !  �  �  ~  6  �  j   �  �  @  �  �  �  	  	  �  �  �  �  W    �  &  �  �  �  �  �  �  �  �  �  �  �  q  ]  J  6      �  �  �  �  t  Q  .    �  �  �  b  =    �  �  �  �  g  H  F  D  D  F  M  P  E  $  k  }  z  h  X  C  !  �  �  �  �  v  U  0    �  v    �   6  1  5  9  ;  9  3  )      �  �  �  �  �  }  J    �  M   �  b  _  \  X  T  O  I  C  <  3  )      �  �  �  �  �  }  g  �  �  �           /  ;  A  I  P  I  -  �  ~    �    8  �  M  ,    �  �  �  F    �  �  E    �  v  <  �  �  i  8  q  Z  B  H  f  b  V  F  7  +  %         �  �  �  Q     �    e  �  �  	  	  	  	  �  �  x  '  �  7  �  -  
  +    �  G  F  D  @  8  ,      �  �  �  �  ^    �  �  L    �  Z     Y  �  �  +  <  N  V  @    �  (  
�  
e  	�  �  �  �  
  H  \  N  ?  6  9  <  :  3  -      �  �  �  �  _  5    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  W  ^  e  l  ^  K  9  %    �  �  �  �  �  �  k  P  3     �  
  5  ;  =  6    �  �  �  m  4  �  �  s  %  �  s    �  $  �  �  �  �  �  �  �  �  �  �  �  �  o  =    �  x  4  �  �  t  p  l  g  `  W  K  ?  .       �  �  �  Y  "  �  �  x  =  �  �  �  �  �  �  �  �  �  �  �  �  �  d  >  
  �  q    �  �  |  v  n  f  ]  T  G  7  $    �  �  �  �  �  Z  ,   �   o  ~  j  `  ]  `  j  x  �  �  v  j  Y  E  (    �  �  Y  /    �  �  x  j  [  K  :  ,      �  �  �  �  �  �  {  b  F  *  ]  Q  D  7  (      �  �  �  �  �  �  �  {  m  ^  P  A  2  w  l  a  R  >  '    �  �  �  �  p  =  �  �  s  =    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  f  i  a  U  C  .    �  �  �  �  [  +  �  �  �  ?  �  �  0  	�  	�  	�  
  
L  
y  
�  
    
�  
�  
�  
n  
-  	�  	#  ?  =  ,  �  t  �  �  �  �  y  d  N  1    �  �  �  ]  )  �  �  >  �  p  �  �  �  �  �  �  �  �  �  �  \  ,  �  �  a    �  �  ]      z  u  m  f  ]  S  I  >  -       �  �  �  �  r  [  D  .  B  |  �  �  �  �  f  E    �  �  g    �  7  �  ,  �  �  �  H  C  =  5  +    	  �  �  �  �  �  f  ;    �  �  >   �   �  �  �  �  �  �  �  v  [  =    �  �  �  �  [  5    �  �  �      �      
  
  �  �  �  �  �    v  ?  �  �  �  �   �                  �  �  �  �  �  �  z  [  =    �    
  �  �  �  �  �  �  u  Z  ;      �  �  �  �  �  �  f  I  i  h  e  \  O  ?  0    �  �  �  �  p  N  *  �  �  �  O    �  m  P  7      �  �  �  �  �  �  �  x  `  A    �  ?  �      �  �  �  �  �  e  H  *    �  �  �  �  i  H  q  �  �  �  �  �  �  �  �  �  }  i  V  B  ,    �  �  �  �  s  >  	  5      �  �  �  f  3  �  �  �  N    �  �  X    �  �  <  O  �  �  �  �  �  n  Q  ,  �  �  ~  /  �  v    �  +  �  ]  N  J  E  A  <  8  4  .  (  "        
    �  �  �  �  �  
�  
�  
�  
\  
  	�  	V  �  �  H  �  �  q  5  �  �  ,  �  �  M  �  �  �  �  �  �  �  �  �  m  L  #  �  �  V  �  �    t  �      �  �  �  �  �    a  >    �  �  �  ~  a  5  	   �   �  [  �  �  �  g  J  '    �  �  ]  !  �  F  �  *  �  �  %  A  �  �  �  �  �  �  v  d  P  6  $    �  �    D    �  �  �  �  �  �  �  �  �  �  �  }  t  k  c  Z  O  @  0  !       �    �  �  �  �  {  F    �  �  �  l  '  �  �  $  �  S  �  �      $      �  �  �  �  �  |  f  V  J  N  Y  b  k  V  >  �  �  �  �  �  �  �  �  �  �    `  5    �  �  t  ^  H  3