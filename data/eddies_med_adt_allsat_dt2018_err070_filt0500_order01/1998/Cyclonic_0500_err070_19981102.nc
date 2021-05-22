CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�n��O�<       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MѨ�   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��
=   max       ;�o       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?E�Q�   max       @FL�����     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vyp��
>     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3         max       @P            �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @��`           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �,1   max       ��1       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��e   max       B0��       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��F   max       B0Kf       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >Fj   max       C��)       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@   max       C��|       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          I       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          M       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MѨ�   max       P��2       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�i�B���   max       ?ٓ��҉       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��"�   max       �t�       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?u\(�   max       @FJ=p��
     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vxz�G�     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P            �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�N�           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?{�u%F   max       ?ٓ��҉     �  \8   <   2      '   -         H   
   /                           H                     4                  2      
                        	      '         2   
               	               $                  <         E      P��P P
�RO�V�O��O��O�+�P��N��PV��N �LOK��O�N�~4O��OnH&Ow�}O5��P`Q�O�@O�MxN�U�N�+MѨ�M��PJ֊Ov�mOcy Od�+O}aN��-O{<�O1}|N0�WO6�@N&��O�*vN�]�O�&N��M�8�Nؿ\NC<O�@bN|��N_�PXN���OPc�Ni�ZOr8�O\��OW�O_<�Ng#NW\Og�1O�H�N�>*O�7�O;�eO�h0N��pO��N�ɉNE�Oy�OvO�N��E;�o��o���
��`B�o�o�t��t��D���T����o��C���C���C���C���C���C����㼣�
��9X��9X��9X�ě��ě��ě����ͼ�����/�+�C��C��C���P��P�#�
�#�
�#�
�#�
�#�
�#�
�<j�<j�@��Y��aG��ixսixսu�y�#�}�}�}󶽁%��%��%��o��7L��\)�������㽛�㽧�{�� Ž�-��-���ͽ��ͽ�
=���������������~�������������|�{w|~�)6BGPQ^_U>)����#<HUanghaZU<,����31)����������"*6CO\bilkhO6*/1<Hanz~|qsqnaUH</./����0b{���{UF0�����<<8/##/<<<<<<<<<<<7@Tmz����������zaT>7~��������wvz~~~~~~~~������������������������
#/6<:9-
�������

 �������GUadnz�������nfHB@CG
)6BO[bgd[OB6'
NOQ[ht��������th[SNN./<HS]^Z\UH?</.)*+-.pz�������������zshipFHQamq�������maWPIHF��������
�����������������������������������������������~�����������~~~~~~~~�����������������������'*1?B6�������)/5HUakv{zndaVH</%#)���������������������������������Uanxz�������zna[RJMU��������������������469=BO[hntoh\OB:6324����������������������������������������`dkmz���������zwmhb`'>@CDQ[t�����tg[0" 'Z[]cgotwyzyyxtmg[ZYZQgoty�����������v[PQ��������������������rtv|�������ytrrrrrr������������������������������������������������� ������������������������������z{|��������{zzzzzzzz������
����������������������������gt���������tmge`^^_gst���������{wtssssss����
#;//#
�������9IUWanz������znaRH=9NY[gjmnptu{tg[WNLJKN}����������������~}}	
#)/0/#
					3<IUaZUI<73333333333��� !##$������[at�����������[SSWX[rtx������������y|trrYgqtw��������tg^XWTYY[]hty|������th\[RQY���������BHIUZagmhaUHHBBBBBBB#-/9<@HUa]K</#!#)08<>AEIJJIA50#!.0<>?@<;0*)+........)1:BGLNIB5) �P[gt����������tg[YUP���������������������
���������������
�#�/�8�?�@�@�<�8�/�#�
�Y�4�%�2�G�Y��������ּۼѼмʼ�����r�Y�;�"����������	��"�;�H�T�a�i�q�j�a�H�;�)�#��������)�1�B�G�t�t�h�b�O�6�)�s�k�g�b�h�s���������������������������s�	�����޾�����	��"�$�/�6�6�5�.�"��	�T�H�8�0�.�,�+�/�H�a�m�z�~���|�y�z�q�m�T��������x�`�A�;�s��������������$�"��t�t�t�o�i�s�t�t�t�t�t�t�t�t�t�t�����ۿ޿ڿпݿ����A�Z�a�a�X�S�K�5��Z�V�Y�Z�g�n�s�������s�g�Z�Z�Z�Z�Z�Z�Z�Z�m�b�`�Z�U�R�T�V�`�m�y�������������}�y�mùìßÜ×ÔÖÜàìù����������������ù�g�f�Z�W�W�Z�g�s�������������������s�g�gìêãìôøþ����������������������ùì����w�y���������������������������������y�s�m�f�h�i�b�`�i�m�y�����������������y������#�/�<�H�P�N�U�V�W�U�H�<�/�#������������������6�B�W�h�r�o�h�B�)����	� ������	�/�9�;�?�H�Q�N�H�;�"���H�@�;�?�H�a�m�v���������������{�m�a�T�H�z�s�n�e�a�_�a�n�zÇÓØÞÞÓÇ�z�z�z�z��۾׾ʾ������������������ʾо׾����ìéàÞßàáìù��ùõìììììììì�f�_�f�l�q�s�w�{�����s�f�f�f�f�f�f�f�f�r�Y�@����4�Y�r�������������ּμ����r�M�E�A�7�6�;�A�M�Z�]�a�f�k�s�~���v�f�Z�MŹŴŵŹ��������������,����������Ź�C�B�6�*�����������*�6�E�Q�X�Q�O�C�I�=�K�Z�\�d�g�s�����������������s�g�Z�I�����������������������������������������������x�m�f�q�����������û˻˻û�������ƧƚƎƁ�w�x�yƁƎƚƝƧƯƳ������ƸƳƧ�H�=�<�0�2�<�?�H�P�U�V�U�H�H�H�H�H�H�H�H����ĿĽĽ���������������� �����������ؾ�ؾ׾Ӿ׾����������������ʾ��������ʾ۾����	�����������ƚƕƎƁ�uƁƇƎƚƧƳ����������ƴƳƧƚ�0�����������������0�I�r�x�o�M�=�0�����������	���������������������O�E�C�6�*�#�*�6�6�6�C�H�O�S�O�O�O�O�O�O�m�g�`�W�X�`�m�y�������������������y�m�m��������$�%�$������������Y�X�L�G�@�6�4�=�L�r�~�����������w�r�e�Y�S�S�S�_�g�l�x�����������x�p�l�_�S�S�S�S�@�8�4�-�4�?�@�G�M�S�M�C�@�@�@�@�@�@�@�@�r�f�^�g�����ʼ����!�$����㼽�����r�I�E�<�<�<�A�I�U�b�n�s�o�n�h�b�U�I�I�I�I�3�*�0�5�A�I�N�Z�a�s�������x�s�g�Z�N�A�3���������������������������������������������������ſƿѿܿݿ�������ݿѿĿ��Ŀ������������������ĿϿѿֿ޿�ݿտѿ��a�U�S�N�U�b�n�{ŅŇŔŝŚŗŔŎŇ�{�n�aŭŠśŐŐŔśŠŭŹ����������������ŹŭE*E*E*E+E7E<ECEOEPEVEXEPENECE:E7E*E*E*E*�_�S�[�V�_�l�n�v�m�l�_�_�_�_�_�_�_�_�_�_���������������(�4�>�>�4�(������<�/���#�,�H�U�n�zÇÕÌÈÎÇ�z�n�U�<����������������� �)�5�)�����t�l�t�}²¿������������¿²¦�������������������ùϹֹٹܹ�ܹ۹Ϲù����	���!�.�S�`�l�y�����{�f�S�G�:�!������������	������	������������E�E�E�E�E�E�E�E�E�FFFF$F'F'FF	E�E�Eٽ��������������Ľнֽݽ����ݽ۽Ž����������������%��������������������M�G�K�S�[�h�xāčĝĚęĝĘđĄ�t�h�[�MĦĠĚėěĚğĦĳ����������������ĿĳĦ�!�������������!�-�:�=�;�:�-�$�! ) Y : L 8 @ 4 z G 3 V , A ` X ^ M ' 6 3 | ; } e f Z C n H Q : 0 D a D T b b N ` � K U & Z 3 | - G U b l H & r c > D K K 3 Q , O n T X X V    I  �  �  �  .  ^  g  K  Q    C  �  v  �  Y  ,    �  �  �  �  �  E  2  I  �  
  :  �    �  �  �  l  �  1  #  W  R  �  n     �  n  �  !  �  �  �  �  Z    V  �  �  �  �  %    �  �  �  �  �  H  _  8    C�aG��P�`�C��<j�Y�����󶽥�T��1�u��1���D����j�,1�o���C���j�@��49X����/�����������-�P�`�8Q�T���49X�,1��1�D���@��}�0 Ž}�ixս�O߽Y��D���]/�Y���^5��\)�q�������O߽�^5��C�����㽓t����㽗�P��7L�� Ž�
=���w��"ѽ����������z�ȴ9����,1��F�B�SB��B6Bf�B�B0��B�)B%kB&�A�ѶB�wBTBϢB�UB֞B�BzjB��B=�A��eB��B�`BF�B7iB $BBv.B��B۞B�B��B�Bc{B!�A�ëB�NB	(�B	��B
4"B^B
�B+/�B �B"ohB 0�B)?B,�5BE�B	ִB
M�B�B�B	OBC�Bx�B&��BĕB
1B
��B	�B��BB�8B&�B%��B%�3Bz1B
9�B�B�B�hB�2BASB��B0KfBh�B&TqB5�A��AB�By�B>�B��B=1B�0BӎB:cB �SA��FB�^B�fB�QB�B֭B��B�"B��B7,B�iB�B�
Bl B ��A���B�gB	A�B	;�B
?dB��B
9`B+A�BEB"��B AB)?�B-?B?B	�`B
(�B��B�YB��BEB��B&G�BC~B
9�B1lB
<�B�BA1B� B�B%��B%�>BA'B	�kB?�A�v@��hA�ЛA�y�A���A]	{A�;1A��A���A��]A��+AkծA�{�A��5Aύ�A�PAnV�A�уA��aA��yA�x�A��AO%A�mXAC"@�vtA>��A�k�A�`iA���A�?�@�d�B!SA��A�iAUw�AX�yB��B
&GB�B {1Am�$B	?�*@�OA@��F@�ZtA�`A���A���AyֺAx�BA�<�A�}C���@�R�A3FAĂA���A��W>FjA��AZS�C��)A%,tA2NA�R$A�U@d.�A��@��A��Aו�A��A[�A���A�j�A��cA�~yA��Ak��A̐�A���A��A�'@Al�mA�gLAցVA�iAA��5Aȝ�ANg�A�xFAB�@�E�A<��A��9A���A��A�}�@�*B�UAĀA��AT�pAY_B�-B
�&BA�B sYAm��B	 [?�{@���@�6A&6A�zA���A�9�Az�)Ax��A�yA���C��*@�
�A5�VAì4A���A���>@A8�AZ�C��|A%A3]�Aۨ�A�{�@d9�   <   3      (   .         I   
   0                            I         	            5               	   3                              	      '         2   
               
               $                   <         E         !   -   )   #   !         M      1                           1      %               ;                                 %      '                        5                                 #      !      !      !                              !         M                                 !      %               ;                                       %                        5                                             !                     O�aO<��O��OF��O�FwO�x�O���P��2N��O�ҜN �LOK��OJL]N�~4O��OU�TN�D�O5��O�]iOD�O�MxN�U�N�+MѨ�M��PJ֊Ov�mOcy OQY�O}aN��-O`ޢN�<�N0�WO%�N&��O6�N�]�Oѱ�Ni��M�8�Nؿ\NC<ON�dN|��N_�PXN�zkOPc�Ni�ZOr8�OH�OW�O_<�Ng#NW\Og�1OZ��N�>*O�QN�@>O�h0N��pON�:N�#NE�OR|�OvO�N��E  �  �  �  \  �  c       �  �      �  +  	    3  �    �    �  {  �  )  g  -  �  �  !  l  	�  �  �  �  K  :  �  s  �  [  �      C  �  i  �  !  �  H  �  `  �    '    �  �    �      �  B    �  �  �t���j�u��t��e`B�t��#�
�#�
�D����`B��o��C���/��C����ͼ�t��ě�����P�`����9X��9X�ě��ě��ě����ͼ�����/�C��C��C����'�P�'#�
�D���#�
�,1�'<j�<j�@���o�aG��ixսixսy�#�y�#�}�}󶽁%��%��%��%��o��7L���T����������9X���{������E���-��"ѽ��ͽ�
=������� ������������������������������)6:BHIHB6)��#/<HPRPJHC</)#���),) ��������#*6CLO\`gjihO6*<Hanz}}{oqnaUH<1102<�����0bn{���{<�����<<8/##/<<<<<<<<<<<NTamz��������maSLJKN~��������wvz~~~~~~~~�������������������������
#'/11+#
������

 �������PUZaknz������zna_UNP)6BO[_fc[OB6)"`hot��������th``````./<HS]^Z\UH?</.)*+-.y���������������zvwyRTamsz�����}zmfaZUOR��������
�����������������������������������������������~�����������~~~~~~~~�����������������������'*1?B6�������)/5HUakv{zndaVH</%#)���������������������������������Uanxz�������zna[RJMU��������������������6:>BO[hiplh[YOB;6446����������������������������������������aelmz{��������zxmicaKNR[gty����ztg[QNGDKZ[]cgotwyzyyxtmg[ZYZTipx������������yZTT��������������������rtv|�������ytrrrrrr��������������������������������������������������������������������������������z{|��������{zzzzzzzz������
����������������������������gt���������tmge`^^_gst���������{wtssssss����
#;//#
�������<KUanz�������zmaTHB<NY[gjmnptu{tg[WNLJKN}����������������~}}	
#)/0/#
					3<IUaZUI<73333333333��� !##$������bgt�����������tgd``brtx������������y|trrY[^mt��������tga[ZWYW[\htv���xth[YWWWWWW���������BHIUZagmhaUHHBBBBBBB#058:>DIF</####05<=??<0-&#####.0<>?@<;0*)+........)/8>BDFB5)P[gt����������tg[YUP�������������������������������������
��#�/�9�<�<�:�/�#����Y�T�M�M�R�Y�f�r������������������r�f�Y�H�;�/�"�����&�/�;�H�T�[�`�g�j�a�T�H�6�5�)�#�"�"�$�)�6�B�O�Q�[�h�n�h�^�[�O�6�s�k�e�h�k�o�z�������������������������s�	������߾������	��"�"�.�4�5�2�.�"�	�9�1�/�-�-�/�;�H�a�m�{���z�w�w�m�a�T�H�9��	�������x�`�C�>�A�s����������#� ��t�t�t�o�i�s�t�t�t�t�t�t�t�t�t�t������������A�N�U�W�J�D�5�(����Z�V�Y�Z�g�n�s�������s�g�Z�Z�Z�Z�Z�Z�Z�Z�m�b�`�Z�U�R�T�V�`�m�y�������������}�y�m��ùìæäßÛÝàêìù�����������������g�f�Z�W�W�Z�g�s�������������������s�g�g��þùóùúþ�����������������������������}�y�z�������������������������������������y�v�m�m�l�m�y����������������������������#�/�<�H�P�N�U�V�W�U�H�<�/�#������������������)�B�M�V�[�S�B�6�)����	��	���"�/�;�B�H�I�K�H�D�;�/�"��H�@�;�?�H�a�m�v���������������{�m�a�T�H�z�s�n�e�a�_�a�n�zÇÓØÞÞÓÇ�z�z�z�z��۾׾ʾ������������������ʾо׾����ìéàÞßàáìù��ùõìììììììì�f�_�f�l�q�s�w�{�����s�f�f�f�f�f�f�f�f�r�Y�@����4�Y�r�������������ּμ����r�M�E�A�7�6�;�A�M�Z�]�a�f�k�s�~���v�f�Z�MŹŴŵŹ��������������,����������Ź����������*�1�6�D�O�U�O�C�6�*���I�=�K�Z�\�d�g�s�����������������s�g�Z�I���������������������������������������������x�p�i�t�������������ûɻɻû�������ƎƎƁ�~ƀƁƎƚƤƧƯƯƧƚƎƎƎƎƎƎ�H�=�<�0�2�<�?�H�P�U�V�U�H�H�H�H�H�H�H�H����ĿľľĿ��������������� �����������ؾ�ؾ׾Ӿ׾���������������������������	������	������ƚƕƎƁ�uƁƇƎƚƧƳ����������ƴƳƧƚ�0���������������$�0�I�b�o�v�o�I�=�0�� �����������������������O�E�C�6�*�#�*�6�6�6�C�H�O�S�O�O�O�O�O�O�m�g�`�W�X�`�m�y�������������������y�m�m��������$�%�$������������Y�L�F�>�=�@�I�L�Y�e�r�~���������~�r�e�Y�S�S�S�_�g�l�x�����������x�p�l�_�S�S�S�S�@�8�4�-�4�?�@�G�M�S�M�C�@�@�@�@�@�@�@�@�r�f�^�g�����ʼ����!�$����㼽�����r�I�F�>�D�I�U�b�n�o�n�l�e�b�U�I�I�I�I�I�I�3�*�0�5�A�I�N�Z�a�s�������x�s�g�Z�N�A�3���������������������������������������������������ſƿѿܿݿ�������ݿѿĿ��Ŀ����������������ĿͿѿҿտݿ�ݿӿѿ��a�U�S�N�U�b�n�{ŅŇŔŝŚŗŔŎŇ�{�n�aŭŠśŐŐŔśŠŭŹ����������������ŹŭE*E*E*E+E7E<ECEOEPEVEXEPENECE:E7E*E*E*E*�_�S�[�V�_�l�n�v�m�l�_�_�_�_�_�_�_�_�_�_���������������(�4�>�>�4�(������<�/�"��#�%�/�<�H�U�a�n�p�k�j�k�a�U�H�<����������������� �)�5�)�����x²¿������������¿²¦�ù����������¹ùŹϹйչҹϹùùùùùý��	���!�.�S�`�l�y�����{�f�S�G�:�!������������	������	������������E�E�E�E�E�E�E�E�E�E�FFFF!FFFE�E�Eٽ��������������Ľн׽սнĽ������������������������%��������������������[�R�O�J�L�U�[�h�t�}āčĒĚĖĎĂ�t�h�[ĦĠĚėěĚğĦĳ����������������ĿĳĦ�!�������������!�-�:�=�;�:�-�$�!  B ; % . ? 5 w G , V , K ` R U p ' -  | ; } e f Z C n A Q : 0 @ a 3 T V b P g � K U  Z 3 | ) G U b k H & r c > $ K J ? Q , M X T N X V    �  �    �  �  <  J  �  Q  I  C  �  �  �  \  �  �  �  �  �  �  �  E  2  I  �  
  :  �    �  �  �  l  k  1  ]  W    �  n     �  �  �  !  �  �  �  �  Z  �  V  �  �  �  �  �    M  �  �  �  �  �  _  �    C  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  ~  �  �  �  �  �  �  �  u  1  �  �  %  �  4  �    g  a  �  >  �  �  	  3  N  r  �  �  �  �  �  Z    �  L    �  H  �  *  O  j  }  �  �  �  �  �  �  k  Q  4  �  �  S  I  )  �  6  y  �  �  +  T  \  W  L  :      �  �  ;  �  k  �  �  *  �  Z  ~  �  �    o  X  <    �  �  �  Y    �  ^  �  U  �  3  a  c  a  [  T  K  @  7  +      �  �  �  �  �  K  �  e   �          �  �  �  �  �  �  �  �  �  ]  *  �  �  �  U  �    
  �  �  
  �  �  ;  �  e  �  �  3  �  �  r  5  �  7   �  �  �  �  �  �  �  �  �  �  }  y  t  p  l  h  b  W  9  �  �  �    ,  L  b  v  �  �  u  b  L  -    �  r  �  W  �     B      �  �  �  �  �  �  �  �  m  Y  B  +    �  �  �  �  �         �  �  �  �  �  �  �  �  d  ;    �  �  I    �  }  _  �  �  �  �  �  �  �  �  �  �  g  7  �  �  P  �  �  #  p  +  #            �  �  �  �  g  K  3  %        %  1  �  �  �  �  �    	    �  �  �  `  (  �  �  M  �  s  �  �              �  �  �  �  �  �  �  �  |  Z  2    �  c  �  �         &  )  *  *  0  3  1  $    "  F  >  9  )    �  �  �  �  b  D  (  7  J  D  8  '    �  �  ^  7    �  �  �  =  r  �  �  �  �        �  �  M  �  J  �  G  I  �  :  y  }  �  �  �  �  �  �  �  �  l  I    �  �  W  �  �    Y    �  �  �  �  �  l  D    �  �  x  Q      �  �  �  ~  /  �  �  t  `  J  0  	  �  �  �  �    d  F  (     �  �  3   �  {  u  p  j  d  ^  Y  O  C  6  *      
              �  �  �  �                            �  �  �  �  )  '  %  #  !                               !  g  `  5  �  �  �  �  �  �    \  *  �  �  y  b  1  �  )  �  -         �  �  �  �  �  �  k  D    �  �  k      �  P  �  �  �  �  �  �  l  T  <  #    �  �  w  &  �  `    �  �  �  �  �  �  �  �  �  �  l  E    �  �  �  U    �  �  �  �  !      �  �  �  �  �  �  �  r  Z  ?  #    �  �  �  �  N  l  c  [  R  H  >  3  '    �  �  �  �  �  v  W  7     �   �  	�  	�  	�  	�  	�  	h  	4  �  �  Y  �  �  0  �  S  �  �  �  w  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  b  H  .      �  �  �  �  �  w  _  D  (  	  �  �  �  X    �  �  t  5  �  �  �  �  �  �  �  z  \  <    �  �  �  m  &  �  ~  &  �  h  K  G  D  @  <  9  5  2  /  ,  )  &  #                 �  �  �  �  �  �  $  :  4  *      �  �  �  e  6  	  �  �  �  l  V  @  /  .  ,      �  �  �  �  N    �  �  K  �  �  k  r  c  V  N  O  O  L  B  .    �  �  �  \    �  V  �   �  �  �    Y  s  �  �    (  %      	  �  �  �  �  �  �  �  [  L  =  .         �  �  �  �  �  �  �  �  �  �  �  �  y  �  �  �  �  �  {  q  g  Q  9       �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  q  _  M  9  $    �  �  �  �  �  �  �  �      	  �  �  �  �  �  Q    �  I  �  Q  �  �  C  A  8  +      �  �  �  �  �  q  Q  8     	  �  �  �  �  �  �  �  �  �  z  q  i  `  X  O  E  <  2  )           �  i  G    �  �  �    �  �  �  �  Z    �  |    �  �     �  �  �  �  �  �  �  �  �  �  �  �  w  Y  :    �  �  �  �  h  !       �  �  �  �  �  {  [  2  �  �  v  &  �  �  p  f  �  �  �  �  �  �  �  �  �  �  �  {  {  |  }  ~  �    x  q  j  H  =  3       �  �  �  k  B    �  �  \  *  �    w  �  �  �  �  �  �  �  �  �  �  �  �  b  7  �  �  �  �  �  ~  Z  #  `  V  L  =  .  ,  .  $      �  �  �  �  �  _  0    �  �  �  �  �  �  �  �  �    i  T  C  1      �  �  �  O    �    �  �  �  K    �  �  �  [  .  �  �  O  	  �  s  &  �  �  '  '  '  &  &  &  &  $  "              %  ,  2  9  ?      
  �  �  �  �  �  �  �  z  Y  1    �  �  /  �  �   �  H  8  2  s  �  �  �  �  �  �  �  g  E    �  s  	  �  �  �  �  �  �  �  �  �  �  �  �  v  j  ]  P  F  A  <  7  3  .  )  �  �     �  �  �  �  �  j  @    �  �  m  .  �  �  V  �    �  �  �  �  �  �  �  �  �  �  �  �  �  T  �  T  �  
  V   �      �  �  �  �  �  y  V  ,    �  �  v  <    �  �  �  �          �  �  �  �  �  �  ~  _  >  	  �  }  0  �  �    z  �  �  �  �  �  �  �  �  l    
�  
J  	�  	   �  �    Z  
  1      5  8  %      �  �  �  �  �  �  �  b  '  �  �  ,    �  �  �  �  �  �  �  �  �  ~  j  S  <  %    �  �  �  �  �  �  �  �  �  ]  #  �  �  )  �  7  �    E  
[  	a  F    �  �  n  M  -  
  �  �  �  �  p  T  0    �  �  R    �  O    �  �  �  �  �  |  ^  ?  "    �  �  �  �  i  2  �  �  Y  