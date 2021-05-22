CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�ȴ9Xb       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N
�   max       P�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��Q�   max       ;o       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @F�33333     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v}\(�     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�d            6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       ���
       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�}�   max       B5�       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�D,   max       B4�P       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =բ   max       C��2       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >5,�   max       C���       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Y       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          F       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N
�   max       P�@�       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��D��*   max       ?��Q�       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��^5   max       �o       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�\   max       @F�33333     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v}\(�     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n/   max       ?���Z�     �  Zh   Y         	                     ;               P            7         D            ,   $   #      3      ,                  	               
   #            	            	   0   	                              )   	   
P�N��@N+��N�.iN��N/pN4lnNk�WN�N1�Pr6N��@O~�O *OаP@MO��N��nOl�O���N� O�B�P>�7N��`N�Q�N!g�O���OÜO�vOu�(P�N�'O�mWN�M�O`sN�sN��}OYq�NʓO�hOۓO
d�O��NqO��O���N�FO�x8N��LO`�N�[yNe�N96PP �Nx�^N�.O���N,PN��$N
�OǌN�C�N?y�N^�O�m�N}�YN���;o%   �o�t��t��#�
�49X�D���T���e`B�u�u��C����㼛�㼼j�ě�����������/��h�o�C��C��C��t���P��P��P����w��w��w��w�''0 Ž0 Ž0 Ž49X�8Q�8Q�D���P�`�]/�aG��aG��aG��ixսq����%��%��%��+��+��C����P���
���
���
��� Ž�-��9X��9X��E���Q���{<����#0b{�����������������������������������������������������������������P[eht�����th_[PPPPPP��������������������qzz�������~zqqqqqqqq��	�����������Xanqrnha_]XXXXXXXXXX[afnz���}zna[[[[[[[[(;HTamz���~ia@;1/,((}���������������}}}}������������

	������
#'<DHJI<3/#
��BHPSMG<����+/<CIUakrqusa^UH<0,+����������������������������������"# ��������46:BEO[_a^[[OB764444�������������Bg|��������g[NB-bhtw����thedbbbbbbbb����������������������������������������`bgt�������������te`?ENt���������tg[PF@?��������������������y����������������ywy��������������������������������������������������������������������������������(),58BN[ghca[ONB5,)(')6<961)%#''''''''''_anzz�����zvnjaa____��������������������>BO[e``\[OBA;;>>>>>>GNR[gt�������tg[ROJG}������������������}��������������������p{���������������{np #)0100&#"        #0<IRkkbUQKA0#����������������������������������������9<IUbosspogcb\U<6439�����������������������������������������������������������������������������;<EHPU[XUH<<;;;;;;;;NVgt������������g[NN,03<EHGA<0-*,,,,,,,,

gt���������vtsg`_\\grt|�����vtrmrrrrrrrr+/<AHJMPH<8/-)++++++SUXadeaURRSSSSSSSSSS��#/9<><4/(#
����{�������������}{{{{{����������������������������� )5FNSSD;5)	qz|�������zpqqqqqqqq� �������������������������s�[�Q�>�<�A�Z��������������������������	�	����������������������ľ������������������������U�O�R�U�U�a�n�q�u�o�n�a�U�U�U�U�U�U�U�U�/�*�#�!���#�/�<�?�F�@�<�7�/�/�/�/�/�/���������������
�
�
� ��������������������
�	��	����"�&�"���������������~��������������������������������z�r�v�z�������������z�z�z�z�z�z�z�z�z�z�U�M�H�F�D�G�H�K�U�V�Y�W�U�U�U�U�U�U�U�U���\�D�%������(�A�Z�s������������������ܻ׻лĻλлܻ������� ���������ʾľ������ľʾ׾�������������׾ʾ�ôîíôù�������������������������ô������%�)�1�6�B�Q�O�N�J�B�=�6�)�%������������(�5�A�Z�g�r�o�_�N�(��
�� �������������
��#�,�/�=�;�;�/��
�y�n�m�`�`�Z�`�m�y���������������y�y�y�y�T�Q�N�G�A�:�:�;�G�T�`�m�y�{���y�r�a�V�T�ּʼμӼ�����!�.�;�B�A�:�.�!�����ֻx�v�l�a�_�Y�[�_�l�x�������������x�x�x�xà×ÍÀ�z�n�e�e�u�zÇÓàçìðóôìà������������������������!�(�(�!��������M�G�J�M�S�Y�f�m�q�p�f�Y�M�M�M�M�M�M�M�M�r�l�k�f�c�f�r���������������������r�r������ھ���������������������������ĿĸĵĲĹĿ������������
�������������	���.�1�4�5�3�,�"��	���_�\�^�_�f�l�o�x�}�����������������x�l�_��ƳƧƚƖƔƚƳ���������������������������ŴŰŹ��������*�6�C�b�x�r�h�O�C�*���ּѼҼּڼ�����������������ּּּּ����������'�4�8�;�9�4�1�'����g�Z�Z�O�P�Y�Z�]�g�s�����������s�g�g�g�gŠşŔŉŇ��|łŀŇŔŠŧŭŭŴŶŭŢŠ���������������ȼ������������������������Z�S�R�Z�Z�e�g�s���������x�s�h�g�Z�Z�Z�Z�������s�k�o�s���������������������������������������������������������������������������������������
��)�2�2�!������r�Y�3�+�'�3�6�3�;�?�Y�e�r�|�~�������~�r���ݹ�������$�!�&����������'�����'�0�J�f�r��������r�f�Y�?�4�'�������úɺֺֺ���ֺɺ����������������-�������!�-�:�D�G�_�l�x�������x�l�-�������������'�C�O�\�sƆƁ�u�\�O�C���ŹŲŭŨŭŶŹż����������������ŹŹŹŹ�û����������ûлܻ����������ܻл��A�8�A�L�N�Z�g�s�y�������s�g�Z�N�A�A�A�Aā�t�[�R�?�1�?�B�O�h�q�tāĊčēčĈăā���������������ĿɿſĿ¿����������������ѿοѿҿܿݿ���������ݿӿѿѿѿѿѿѾM�M�A�?�A�C�M�Z�^�`�Z�U�M�M�M�M�M�M�M�M�I�0��� ������/�=�I�_�j�q�p�n�b�I���������(�.�+�(���������E*EE*E,E7ECEPE\E^EbE\EPECE7E*E*E*E*E*E*´ª¥«²¿����������
��
����������´�#� ����#�0�4�<�=�<�0�#�#�#�#�#�#�#�#FFFFF!F$F1F=FBFAF=F5F1F$FFFFFF�����������ùϹ͹ù���������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E
ED�D�D��<�7�8�<�H�H�U�a�n�u�r�n�a�\�U�H�<�<�<�<�[�S�O�J�O�[�h�o�t�y�t�h�[�[�[�[�[�[�[�[�����)�6�>�A�6�)��������������ùàÓ�ÁÓàìù������������뿸���������������Ŀ̿ʿĿ����������������!���!�.�:�G�S�`�e�e�`�S�G�:�.�!�!�!�! X c . [ e ; 8 Q K � z 4 9 a ] 0 4 ; G H ; ] R G = X = T I N m D  . D T b 5 , : G 7 c H C q Y . r O > Z + T > b < m D W n V X 6 ~ < l       �  3  �  �  Q  [  ~  B  �  l  �  5  �  y  �  �  �  b    �  C  �  �    M  =    `  .  �  �     �  e  *  �  �  �  �  �  2  �  G  ;  �  �  $  �    �  �  R    s  �  N  K  �  ;  �  �  Q  f  <  �  �� że`B���
��C����㼋C���o��C���o���㽕���h�+�+�,1���`��w���o��49X�q������,1�H�9��w���T��t���hs�T����E��m�h����L�ͽaG��Y��H�9�e`B�P�`��\)�����%��7L�y�#��E����T�q����+��+��-��O߽�\)��t���xս��P�� Ž��ͽ����������S����ͽƧ�ě����ȴ9����B&��B�B5�B(iB�HB1�B $B1B[fBԋA�}�B��BI�B��B��B=?B��B+]�B.7�B-��B��B&B	�B:�B Z�BH�B �?B	��B!�B �dB�B̋B�nB�B��B}B�B	WB�nB	؊B ��B![�B*boB%H=B%z�BE�B��B'B�B�QB4�B�LBTB
�B&/�B��B
PB
)LB��B�B.�B
�5B�B�xB�B�`BڮB'ɀB<pB4�PB�{B>�B<�B =9B@zBF�B�'A�D,BK�B@B_�B�B:�B=�B+=�B.A�B-�8B��B�`B�XBK�B @4B�^B ÕB	�B �'B �0B�gB�,B��B��B�%BB�B:HB7gB�qB	�EB FuB!>_B*��B%@}B%5�B��B��B'�B��B?PBB�0B|&B	ͣB&BB"B
8�B	�vB�B�8B��B
�yB�4B¯B@;B��B�A���A�a AL�OA�@GA�X:A��A�h�AH�vA�i�A�A��t@�-AT(9A� -A��$A��A�J�Al�BAh�A��@�ӮA�B�A���@ڼ�@�r�AV?�A�IA[\@��&B6�A��A�@��6A���A��y@�.�A���A�A�A���A�ܯ?�/Q?O�/@�l'@8}�@���B YA���@��WA��Aڒ�Av+	A~U�A=��B
c�A4�C��{A�I�A��C��2=բC�/�A�p|A�%fAֹ�A�u�Av�A�sA��A���AL��A�w�AnA�`.A�r�AJ��A��HA�{#A���@��AS�A΂�A�z�A�X�A��Am=�Af3HA	5@��EAʝ�A�O�@�M�@��$AU�A�lA[�@���B�4A��+A��@���A��A�l@�<A��QA�[5A���A���?ף?M�@ݱ@4T�@�rlBJ�A�r@��A��1A�|�Au~�A~��A<�FB@A4(C���A���A�|�C���>5,�C�7�A��Aۂ�A�\�A�}Aw*�A�p   Y         	                     ;               P            8         D   	         -   $   #      3      -                  	                  $            
            
   1   	                           	   *   
   
   F                              /               )            #         1            '   !         1                              #      !      )   !                        +                                 '         C                                             !            #                                                               #               !                        +                                 #      P�@�Nd)iN+��N�.iN��N/pN4lnNk�WN�N1�O��:N*BNn�OIO��O�O��N��nN�'�O��N��O�B�O��uN��`N�Q�N!g�O�b�O��pN�p�Ou�(O>}�N�'OI�N��YN�O�N�sN��}OYq�NʓOG��OۓN���OU<�NqO�y~O���N�FOtV�N��LO��N�[yNe�N96PP �Nx�^N�.O6�yN,PN��$N
�O��N��nN?y�N^�O���N}�YN��b  �  �  I  �  6    {  �  �  �  	�      �  �  
N    �  �  !  �  #  [  �  �  �  �  �  �  �    �  �  �  �  3  �    �  >  �  $  �  �  �  A  q  �  L  �  �  �  �  �  �  s  H  �  v  =    -  y  �  
s  �  ⻃o�D���o�t��t��#�
�49X�D���T���e`B�o��1��j���
���
�t��ě�������/�o���o�e`B�C��C��t��,1�8Q�0 Ž��y�#��w�T���#�
�0 Ž'0 Ž0 Ž0 ŽY��8Q�L�ͽP�`�P�`��o�aG��aG��e`B�ixս�o��%��%��%��+��+��C����T���
���
���
��1��-��-��9X��^5��E���^5��#0b{������{<
���������������������������������������������������������������P[eht�����th_[PPPPPP��������������������qzz�������~zqqqqqqqq��	�����������Xanqrnha_]XXXXXXXXXX[afnz���}zna[[[[[[[[GHSYamz����~zjaTH@CG�����������������������
	�����������



������	
%/<BHIH<7/#
	�)?IMLHB6�����+/<CIUakrqusa^UH<0,+����������������������
������������� !������;BHO[\_[[OKB;7;;;;;;�������������>N[l~�����tg[NFC>:9>bhtw����thedbbbbbbbb����������������������������������������lxz������������yjdflINZgt��������tg[XLHI��������������������y����������������ywy��������������������������������������������������������������������������������-5:BN[[_^[NKB5.*----')6<961)%#''''''''''_anzz�����zvnjaa____��������������������>BO[e``\[OBA;;>>>>>>T\gpt��������wtgf[YT}������������������}���������������������������������������� #)0100&#"        	#0CIRRKIC<0#
	����������������������������������������<IUbdlonnmeaUI?<765<�������������������������������������������������������������������������������;<EHPU[XUH<<;;;;;;;;NVgt������������g[NN,03<EHGA<0-*,,,,,,,,

ggt���������tqgfebcgrt|�����vtrmrrrrrrrr+/<AHJMPH<8/-)++++++SUXadeaURRSSSSSSSSSS��#/41/&#
����}�������������~|}}}}�����������������������������!)5:DKQQC:5)qz|�������zpqqqqqqqq�����������������s�_�U�B�C�V�s����������������������������������������������������������������������ľ������������������������U�O�R�U�U�a�n�q�u�o�n�a�U�U�U�U�U�U�U�U�/�*�#�!���#�/�<�?�F�@�<�7�/�/�/�/�/�/���������������
�
�
� ��������������������
�	��	����"�&�"���������������~��������������������������������z�r�v�z�������������z�z�z�z�z�z�z�z�z�z�U�M�H�F�D�G�H�K�U�V�Y�W�U�U�U�U�U�U�U�U�Z�X�5�(�����(�5�A�N�T�g�s�}��s�g�Z�ֻܻܻܻ����������ܻܻܻܻܻܻܻܻܻܾ׾ԾʾžʾԾ׾��������׾׾׾׾׾�öïïöù�������������������������ö�����!�&�)�3�6�B�P�O�M�I�B�<�6�)�"������������(�5�A�T�X�V�N�@�(���
�� �������������
��#�,�/�=�;�;�/��
�y�n�m�`�`�Z�`�m�y���������������y�y�y�y�G�C�<�>�G�T�`�m�u�y�}�y�n�m�`�T�G�G�G�G��ּҼҼ׼������!�.�8�?�>�7�.�!����l�c�_�[�^�_�l�x�x���������x�l�l�l�l�l�là×ÍÀ�z�n�e�e�u�zÇÓàçìðóôìà���������������������	����������������M�G�J�M�S�Y�f�m�q�p�f�Y�M�M�M�M�M�M�M�M�r�l�k�f�c�f�r���������������������r�r������ھ�������������������������ĿĺĸķĸĿ����������
�������ؾ�����������	��"�)�-�/�,�%�"��	���l�e�_�^�_�a�i�l�x�x�����������������x�l��ƳƧƚƖƔƚƳ�������������������������������������������������������ּѼҼּڼ�����������������ּּּּ��������������"�'�1�4�5�4�(�'��g�\�Z�Q�S�Z�g�s�����������s�g�g�g�g�g�gŔŋŇŁ�ŇŇŔŠŤŪŭŲŴŭŠŔŔŔŔ���������������ȼ������������������������Z�S�R�Z�Z�e�g�s���������x�s�h�g�Z�Z�Z�Z�������s�k�o�s��������������������������������������������������������������������������������������������������r�Y�3�+�'�3�6�3�;�?�Y�e�r�|�~�������~�r���������������������������@�4�'��'�4�@�M�Y�f�r��������r�f�Y�M�@�������úɺֺֺ���ֺɺ����������������_�S�F�!�
��!�)�-�:�F�S�_�l�x�|�|�p�l�_�������������'�C�O�\�sƆƁ�u�\�O�C���ŹŲŭŨŭŶŹż����������������ŹŹŹŹ�û��������ûлܻ������������л��A�8�A�L�N�Z�g�s�y�������s�g�Z�N�A�A�A�A�[�W�O�B�6�5�6�B�O�[�h�i�sāĂā�|�t�h�[���������������ĿɿſĿ¿����������������ѿοѿҿܿݿ���������ݿӿѿѿѿѿѿѾM�M�A�?�A�C�M�Z�^�`�Z�U�M�M�M�M�M�M�M�M�I�0��� ������/�=�I�_�j�q�p�n�b�I���������(�.�+�(���������E*EE*E,E7ECEPE\E^EbE\EPECE7E*E*E*E*E*E*¿¾²«²¿������������� ������������¿�#� ����#�0�4�<�=�<�0�#�#�#�#�#�#�#�#FFFFF!F$F1F=FBFAF=F5F1F$FFFFFF�����������ùϹ͹ù���������������������D�D�D�D�D�D�D�D�D�D�D�D�D�EEED�D�D�D��<�8�9�<�H�I�U�a�n�r�q�n�a�Z�U�H�<�<�<�<�[�S�O�J�O�[�h�o�t�y�t�h�[�[�[�[�[�[�[�[�����)�6�>�A�6�)���������������ùàÓÁÃÇÓàìù���������	������������������Ŀ̿ʿĿ����������������!���!�.�:�G�N�S�^�S�G�:�.�!�!�!�!�!�! M S . [ e ; 8 Q K � d : - ^ \  4 ; ( 7 A ] ! G = X 4 E A N 4 D  & : T b 5 , 3 G 5 Y H 9 q Y & r J > Z + T > b @ m D W Y S X 6 v < Y    �  �  3  �  �  Q  [  ~  B  �  o  3  j  t  k    �  �  �  �  �  C  �  �    M  �  "    .  �  �  B  �    *  �  �  �  �  �  �  �  G  X  �  �  �  �  [  �  �  R    s  �  �  K  �  ;  E  �  Q  f  �  �  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �  �  �  M  ,  �  �  m  P  I  	  �  o    �  �        �  �  �  �  �  �  �  �  �  d  >    �  �  �  _    �  �  2  I  G  E  B  @  >  ;  4  +  "         �   �   �   �   �   �   �  �  �  �  �  �  j  S  <  a  �  �  �  �  �  �  �  �  F  7  (  6  5  3  1  .  (  !    	  �  �  �  �  x  N  "  �  �  �  U      �  �  �  �  �  p  J    �  �  �  n  E  !  �  �  �  �  {  x  u  q  m  c  Y  N  A  -      �  �  �  �  �  e  G  )  �  �  �  �  �  �  �  �  �  �  x  h  X  G  6  &  '  ,  2  7  �  �  �  �  �  �  �  �  �  �  v  m  d  U  <  #  
   �   �   �  �  �  �  �  �  �  �  �  �  �    "  0  6  <  B  H  M  R  W  �  �  	  	W  	  	�  	�  	}  	q  	f  	B  	  �  �    z  �     @  e  �  �  �  �  �  �  �  �      �  �  �  �  �  {  H  ,  o  K  �  �  �  �             �  �  �  c  )  �  �  h    �  >  �  �  �  �  �  �  ]  4    �  �  |  s  �  ~  [  6    �  Q  �  �  �  �  �  �  �  \  4    �  �  �  u  <  �  �  D  �     	�  
  
@  
N  
I  
>  
&  	�  	�  	�  	D  �  �    �  �      �        �  �  �  �  �  �  o  U  :  $        �  �  �  �  {  �  �  �  �  �  �  �  �  �  �  �  �  �  |  s  h  \  P  D  8  �  �  �  �  �  �  �  �  �  �  �  x  l  `  T  I  ?  :  6  2      !    �  �  �  m  6  �  �  �  d  +  �  �    �  Z   �  �  �  �  �  �  �  �  �  �  �  �  n  U  ;    �  �  �  H  �  #    �  �  �  �  �  �  �  l  =    �  �  D  �  �  a  �  �  c  �  �    0  G  Z  N  3    �  k  	  �  [    �  �      �  �  �  �  �  �  �  �  �  �  �  r  [  2  	  �  �  �  �  �  �  �  �  q  `  V  I  <  -      �  �  �  H    �  �  `  %  �  �  �  �  �  �  ~  o  _  N  >  -      �  �  �  �  �  �  R  y  �  �  s  ^  B    �  �  p  *  �  �    �  @  �  �    R  t  �  �  �  �  �  �  g  E  !  �  �  �  d    �  L  �  �  �  �  �  �  �  �  �  �  s  M  "  �  �  U  �  �  &  �  �  �  �  �  �  �  �  �  �  f  J  .    �  �  �  g  9  
  �  �  �  �  �  �  Q  �  �  �          �  �  �  .  �  _  �  +  �  �  �  �  �  �  |  T  +    �  �  [    �  �  ?  �  �  ?  a    9  R  v  �  �  �  �  p  G    �  �  <  �  �  �  �  �   X  �  �  �  �  �  �  �  �  �  �  �  �  �  g  K  -    �  �  �  �  �  �  �  �  �  �  u  b  K  0    �  �  y  >  �  g  �  �  3  3  3  2  0  ,  &      �    ;  8  0  '           �  �  �  �  �  �  �  s  `  L  8  #    �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  r  [  2    �  �  �  c  3  �  �  �  �  �  �  v  g  U  C  0      �  �  �  j  ;     �     !  $  &  -  8  =  ;  +    �  �  �  z  V     �  �  v  }  �  �  �  �  �  q  X  9    �  �  \    �  �  k  7  �  �  �  �  �  �      !  #      �  �  �  w  N  $     �  �  /  �  �  �  �  �  �  �  �  �  i  F    �  �  �  `    �  M   �   �  �  �  �  �  �  p  [  E  /      �  �  �  �  �  �  z  p  g  �  �  �  �  �  �  �  �  �  �  \    �  �  B  �  ~  �  D  a  A  ?  5    �  �  �  �  g  :    �  �  �  I  �  $  �  �  A  q  g  ^  T  J  ?  0         �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  r  d  U  E  5  %    �  �  �  �  �  �  |  q  L  @  5  .  (        �  �  �  �  �  �  �  �  �  �  �  �  G  _  R  t  �  �  m  M    �  �  d  6  
  �    �    p  �  �  �  �  p  V  :      �  �  �  �  �  �  k  V  B  0      �  �  �  �  �  �  �  y  l  e  ^  Y  [  \  a  i  q  c  M  7  �  �  �  �  �  �  �  �  �  ~  g  O  5    �  �  �  r  8  �  �  i  9    �  {  ]  ^  %  �  �  u  7  �  �  e  �  U  �  �  �  �  �  �  �  �  �  y  b  I  1    �  �  �  �  �  }  c  J  s  c  N  8  $    �  �  �  �  r  K  (    �  �  �  �  >  �     0  2  9  D  G  C  3    �  �  �  P    �  �  S  �  z    �  �  �  �  �  �  �  �  �  �    w  n  a  K  6  !     �   �  v  u  o  g  \  K  0    �  �  �  U    �  �  w  C    �  �  =  :  6  2  .  *  &  !              �  �  �  �  �  �  �        �  �  �  �  [  &  �  �  �  V    �  X  z  �  �  ,  ,  +  (  $  #    �  �  �  T    �  �  2  �  �  9  �  v  y  x  v  g  X  J  <  .  $      	  �  �  �  �  �  �  r  Z  �  �  �  �  �  �  �  �  {  t  m  d  [  V  R  T  Y  g  �  �  
`  
m  
j  
=  	�  	�  	\  �  �    �    �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  `  T  G  8  )       �  �  �  �  �  �  �  �  �  �  �  �  s  i  h  j  v    i  S