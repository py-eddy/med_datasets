CDF       
      obs    P   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��-V     @  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�t�   max       Pbw;     @  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���#   max       <T��     @   ,   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @F���Q�     �  !l   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @viG�z�     �  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q�           �  :l   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�9�         @  ;   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       <o     @  <L   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4�|     @  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��I   max       B4ƨ     @  >�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >P�U   max       C�%�     @  @   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@   max       C�'�     @  AL   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          :     @  B�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          /     @  C�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -     @  E   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�t�   max       P?�     @  FL   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���+j��   max       ?�^5?|�     @  G�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���m   max       <T��     @  H�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @F���Q�     �  J   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @viG�z�     �  V�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q�           �  c   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @��`         @  c�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�     @  d�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?r�s�PH   max       ?�Ov_ح�        f,                                     "                                    +         ,   2            6   	   :            6         	            7                     1         	         	               &      
         
                     #   '            NYȺO{��O*��N��dOE�N�N�O�N
O;O�]N��fO��kN.^�Oy<�O�aO�@N��&N�RtN[k�N�hN��RO7VO�J�O�-Nǌ�OjGgPe�Pbw;N/4O�N��'P1WN��P!یN�kwO��yO���P�yN��gO�dN�TQO�[vO]N���O���OqN��N�FqOb�N��N�x.Odl�N���OM �NG<�M� O8ՎNz�NwnPO�kNUևN�SOʨ�Ni�wO:z�O���N�O!�O��VO�N�QO�HN"�PM�t�O%P�O?��N���N���N;�+N�T�<T��<T��;�`B;�`B:�o:�o%   %   %   %   ��o�o��o�ě��t��49X�e`B�e`B��o��C���t����㼣�
��1��9X��9X��j��j��j��j�ě����ͼ��ͼ��ͼ�/��/��/��/��/��`B��`B��h�������o�+�C��\)�t��t��t���P��w�#�
�#�
�#�
�#�
�'''0 Ž0 Ž49X�@��D���D���H�9�P�`�Y��]/�u��o�����+��t�������1���ͽ��#��� 
!
����������	
#$/<QU\_XUH<#	�')*,))�����MOW[ahntth[ROMMMMMotx{������������tqoo��������������������������������������������	

�����������������������������#'+6BFPQ[msnh[B,!16BMOXYOB64111111111�����%+-%
������^ahnz����zna^^^^^^^^/<HUXaif^UOA<4#������������������������������������������������������!#/<HUUUNTLHE<://#!!��������������������NOV[[hhiiih\[OBGNNNN��������������������@BN[dgr|}ttg`[NB?78@)5Nbt�������tNB)!")�����������������������������������������������������������������
')&���������	#0Ibn{����{<0
	����������������������*6C\{uk\6���)06:;6)��������������������V[ahitlh[XVVVVVVVVVV�������
!'& �������\almpz����zmaZ\\\\\\����������������������������������������������������������������������9<@HOU\abika^UHA<569<<IUYbnonjdbWUI=<9<<)+6BJOSROJB6)ABDN[_c]ba[NB?==<=AA`git~����utgd```````(/6BO[ht����zth[L4.(����������������������������������������)25BJLB52)rz����������������zr��������������������7<<GHUaca`XUH><:7777?HUaiqwz~zsneaUHC?>?���	

 ��������Z`afgpt}��������~h[Z"')-,+,+)45BEDB95/14444444444������


��������#0340)#{��������vxx{{{{{{{{�����������������������������������������������������������������������������mnrp{������{wnmmmmmmyz{�������������zzyamz�����������zma\^a����������������������������������������KUanz���������znaURKkt���������������tik�������������������������� ������������#*070#UU^bbeb[VUQSUUUUUUUUbgjt����������vtgebb
)27;?BKWWNB5'
��������������������st�������������tssss9<?HLTHC<:6599999999#'/00/.%# �������������������������4�(����!�'�4�7�A�M�Z�]�^�_�^�Z�M�A�4������$�+�0�=�I�J�T�U�I�I�=�0�$���e�`�Y�V�Y�[�e�l�r�~�������~�x�r�e�e�e�e�@�7�4�'����'�4�M�W�Y�f�l�l�i�_�Y�M�@���������ʾ־׾۾׾ʾ��������������������������(�)�2�(�����������{�u�{�|ŇňŔŠŭŹ��ŽŹŴŭŠŔŇ�{�{�������������������������������������m�`�T�.��#�.�;�G�R�m�����������������m�U�K�L�U�V�a�n�z�v�n�n�a�U�U�U�U�U�U�U�U�����������������������������������������U�M�H�G�F�F�H�U�W�Z�Y�V�U�U�U�U�U�U�U�U�)�"�$�!�����)�6�B�G�O�[�c�[�M�F�6�)ÓÇ�y�q�q�}�ÇÓàëù��������ùìàÓ�(�$��(�5�O�Z�s�������������s�Z�N�A�5�(�ݿտѿ̿Ͽѿݿ���� �������ݿݿݿ��
�����	�
�
��#�/�0�/�*�#����
�
�M�C�@�=�@�M�Y�f�j�i�f�Y�M�M�M�M�M�M�M�M�;�8�/�"� ��"�/�;�H�K�T�U�]�T�H�;�;�;�;�r�j�Y�X�X�U�Y�e�o�r�~�����������~�z�~�r������"�*�6�C�O�P�T�O�M�I�D�C�6�*����������	����.�4�5�����	����Ěčā�t�h�R�H�Q�[�hāėĚĦķĳīįĦĚ�z�u�r�y�zÇÓÕàì÷íìàÓÇ�z�z�z�z�`�V�G�?�A�G�T�`�m�y���������������y�m�`����Ⱦ��������ʾ׾�����"�;�E�D�.������s�g�T�N�H�@�7�;�Q�s������������������������������������������G�?��������"�.�6�;�@�G�O�V�[�Z�T�G�������������������������������������������h�_�l�����лܻ���������лû����ù��������ùϹչչϹùùùùùùùùùü��������������ּ���!�.�:�>�:�4�$���㼽ĳĭĦĦĦĪĳĳĳĿ������ĿĳĳĳĳĳĳĿĻĺĿ�����������'�!������������Ŀ��ż��������������������������������s�g�N�A�(���5�Z�s�������������������s��~�z�~������������������������������	�������������	���"�0�3�.�+�"��	���������������������ʼͼּ׼ּܼռʼ����S�J�M�X�g�s���������������������s�f�SĳĲĳĵľĿ����������������������Ŀĳĳ��������(�)�5�8�=�5�)���������s�o�j�h�g�b�a�g�s���������������������(� �����(�.�5�A�N�Z�b�Z�W�S�M�A�5�(�׾׾־оʾ��������������������ʾ׾׾׾������������������������������������������H�D�=�;�A�H�T�a�m�z�������}�z�q�a�Q�J�H��
������)�)�)�)�%� ����������z�z�m�k�a�b�m�t�z���������������������Ϲù¹ȹϹܹ��������������ܹϺ����������!�'�-�!������������ŠŔŇ�n�h�b�P�b�c�nŇŔŠŭųŻŹŭţŠ�h�e�[�O�J�O�[�h�tāĆăā�t�h�h�h�h�h�h�)�%�$�)�6�9�B�H�B�6�)�)�)�)�)�)�)�)�)�)�N�I�B�8�?�B�L�N�[�g�t�~�t�g�[�N���������������������������������������������������
�����
�������������������#�������������
��1�@�G�Y�^�[�U�I�0�#������������������������ֺκɺúɺҺֺ�������������ֺֺֺֺY�L�@�<�>�I�Y�n���������Ⱥ̺ɺ������~�Y�@�<�4�'��'�(�4�@�M�M�Q�M�C�@�@�@�@�@�@ƳưƧƚƋƁƀƁƎƚƧƳƽ������������Ƴ������������������������	�� � ������������������������������������������������6�4�+�.�6�C�O�\�h�p�s�h�g�\�O�C�6�6�6�6�Ŀ����������������ѿݿ��ܿۿ�߿ҿѿĻ����������}��������������������������������������������������������ǾʾϾ˾ʾ����������������$�0�<�4�0�(�$�"��������������ûɻлڻڻлͻû�����������������������������������������������<�2�/�,�)�)�-�/�<�H�U�`�`�X�U�T�L�H�<�<ÓÇ�}�~ÇÓàìóù����������ùìèàÓ���������������������������������������������ÿú����������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� v K B < 6 6 8 < a D C , � U < ~ B p   d Z E r R T ) ] ; h x 0 K 4 B K K B S C , d 7 E 1 : 9 F q s X _ 3 t M � f ! : Z ` 3 9 P @ N F [ 4 r a Y ; t � D � B _ H R  �  %  }  �  �    1  :  3  6  �  H  �  '  h  �  �  �  d  �    �  ]  A  �  �    �  D    �  �  5    �  a    }    A  �    Q  �  :  d  �  �  g  �  �  �  �  �  �    |  �  �  ^  h  �    s  �  \  .  -  u  �  �  =  c    {  �  �  �  b  �<o�e`B�#�
�T������o�D����1�ě���/��o�\)�#�
��h��w�+��t����㼣�
��9X�ě����8Q콁%�t���P��+��������0 ż�/���w�+��\)�<j�#�
���
�T���8Q�t��D���<j�t���{�0 Ž���P�@��,1�49X��{�,1�L�ͽH�9�49X�q���D���8Q콍O߽@��P�`���T�H�9�e`B���P�L�ͽq����%��C��u���P��7L��C����ͽ�S���j��^5��S��t�By1Bt�B��B�CB�B4�|B�BE�B�\B�WBOgBY/B��B}�Bz�B�QBQ�B�MB#B8<B!�jB��B��B�B�B�SB��B&��B)��B0�B/(B�DB�;B-qA���B�B�B�$B��Bv�B'.B��B&�B	�BW<Bu�BY�BPBIB�_BR�B��B#�<B	�BʘBshB
�B%#�B
W�B��B�QB?�B@�B(�vB ��B ��B��BD�B��B�}B ��B�~B%8lB'�=B
e�Bx}B�B
�jB��B�#B7�BEB�BH�B��B4ƨB�DBF�B��B��BHBC<B�B��B�7B�GBL�B>nB"��B�B"3B9B	�pB�rB�VB��BB�B&��B)kB/�BDmB>�B�fB-�,A��IB@ BGB@0BœB@>B'�B�%B�B	S�B�BW�B@�B<�B�RB�ZBAQB�B#�sB	�qB�BtHB91B%<�B
@{B�KB�B=�B@zB(�/B ��B ?�B/_B@B
B =�B ��B�-B$��B'�PB
��BA�B
��B9�B��B��A/��A;%5B
(�?��@քSAPmAA5�HA�Y\A�j�An�Aƈ�A��AŞ�A׼[A�=�A��A~��A�?2@اA��>?�ޛB 2�A\�A��A�Y�Aj\�AV��A��	A���Ac�A��Z@��=>P�UA3A���A�A�?A���@�UA\S@���AE��A�\xA�aA��SA�ʒAOtA���A�A�3WA��`?WC@^�|A���A۵A׀@A���A"�TA��A���??S�@Eο@�@�J�BڢA��A���B6_Az�@���AL�;B	'�@��OA0A��A�1BX�A�e`C���C�%�A.��A;(�B
��?���@�dIAO��A5A�u�A�u�Aml-A�~HA�[�AŦ6A�~�A�[�A�t�A}��A�y
@�Q|A�\�?��B =RA\ƴA܇GA��Aj�HATA��GA���A^o�A��@�$�>@A��A�A塓A�yA�^ @��A\��@�LAGRA㨚A�tA���A�AiAN&TA���A�}�A���A��y? 9@a�/A�Q7A�|A���A��MA"9�A爪A�u�?J�r@F?q@g�@�MtBxmA��A�6�BD�Ax�@��kAKiB	E�@��cA.�pA�
Aʴ�B7RA��CC��C�'�                                     "                                    +         ,   3            6   	   :   	         6         	            7                  	   2         
         	            	   '      
                              #   (                                          )                                       !            +   /      +      +      -            )                     '                                             #         #         #                                                                                                                           !   -      +      )      +                                 #                                             !                  #                                          NYȺO#�O*��N�*FN�4�N�N�N��N
O;O �N=j"O�?N�N�4lOk��N�d N��&N�RtN[k�N�hN��RO7VO�JN���Nǌ�N�)�O�yoP?�N/4O�N��'P%�N��P�TN�kwO��yO���O��N�jJN�(�N�TQOmN�N���OǞ�N�9N��N�FqOb�N��Nh�}OMi�N���N�lNG<�M� N���Nz�NwnPO�]XNUևN�SO�L�Ni�wO:z�O���N�O!�O��VO�N�QO�HN"�PM�t�O%P�N��N���N���N;�+Nxuv  �  	  *  F  �  !  �  �  .  s  h  ?  �  {  �  �    {  �  z  >     �  -  b  1  8  5  �  �  �    �  z  �  Z  �  8  �    /    \  �    �  \  �  �  �  q  	�  V  �    $  �  $  '  9  �  �  �  �  	  y        �  b  +  �  �  	1  6  	  V  �  	<T��;�`B;�`B;ě��t�:�o%   ���
%   �e`B���
�u���
��C��u��t��e`B�e`B��o��C���t����㼬1��P��9X��/�C���`B��j��j�ě��������ͼ�h��/��/��/�,1���o��`B���o�����C��+�C��\)�t���P��w��P�,1�#�
�#�
�D���#�
�'49X�'0 ŽH�9�49X�@��D���D���H�9�P�`�Y��]/�u��o�����+�� Ž�����1���ͽ��m��� 
!
����������#/<JUWXUHE</#!�')*,))�����NOY[^hnt}~th[RONNNNNstu�������������utss�������������������������������������������


�����������������������������06;BJO[_ehhe^[OB:52066BIOUVOB=6666666666���

��������`alnz����zna````````#/<HOLH@<5/&#������
�������������������������������������������������!#/<HUUUNTLHE<://#!!��������������������NOV[[hhiiih\[OBGNNNN��������������������@BN[dgr|}ttg`[NB?78@#)5BN_t������tNB)$ #�����������������������������������������������������������������
���������#0Kbn{����{n<0
����������������������*6C\{uk\6���)06:;6)��������������������V[ahitlh[XVVVVVVVVVV�����	%$��������\almpz����zmaZ\\\\\\��������������������������������������������������������������������������9<<FHU[adaaYUIHH<899<<IUYbnonjdbWUI=<9<<)56BINROHB96)=BN[\a[][ZNB?==<====`git~����utgd```````.8B[t����wth[OG>72/.����������������������������������������)25BJLB52)rz����������������zr��������������������8<=HHU`]UUH<88888888@HTUagnotxwpnaUHD@?@���	

 ��������fgkt}������~trgdddff"')-,+,+)45BEDB95/14444444444����


�����������#0340)#{��������vxx{{{{{{{{�����������������������������������������������������������������������������mnrp{������{wnmmmmmmyz{�������������zzyamz�����������zma\^a����������������������������������������KUanz���������znaURKkt���������������tik�������������������������� ������������#*070#UU^bbeb[VUQSUUUUUUUUbgjt����������vtgebb)+155)��������������������st�������������tssss9<?HLTHC<:6599999999#&//0/-$#!�������������������������4�/�*�(�-�4�>�A�M�S�Y�Z�[�[�Z�Z�N�M�A�4������$�+�0�=�I�J�T�U�I�I�=�0�$���e�b�Y�W�Y�\�e�m�r�~�������~�v�r�e�e�e�e�M�D�@�7�4�4�4�@�J�M�Y�b�f�g�f�e�Y�U�M�M���������ʾ־׾۾׾ʾ��������������������������(�)�2�(����������ŇņłŇōŔŠŭŹźŹŸŭŭŠŔŇŇŇŇ�������������������������������������m�b�`�\�`�a�i�m�y�������������������y�m�U�U�P�U�Y�a�n�v�q�n�a�V�U�U�U�U�U�U�U�U�����������������������������������������U�Q�H�H�G�H�H�U�V�Y�X�V�U�U�U�U�U�U�U�U�6�-�*�+�-�6�B�G�O�S�O�L�B�>�6�6�6�6�6�6ÓÇ��x�w�zÅÇÓàãñù������ùìàÓ�Z�R�N�Z�[�g�s�����������������}�s�g�Z�Z�ݿտѿ̿Ͽѿݿ���� �������ݿݿݿ��
�����	�
�
��#�/�0�/�*�#����
�
�M�C�@�=�@�M�Y�f�j�i�f�Y�M�M�M�M�M�M�M�M�;�8�/�"� ��"�/�;�H�K�T�U�]�T�H�;�;�;�;�r�j�Y�X�X�U�Y�e�o�r�~�����������~�z�~�r������"�*�6�C�O�P�T�O�M�I�D�C�6�*�������������	��"�.�3�4�����	��ā�t�h�]�[�V�[�a�h�t�{āčęĚěĚčĊā�z�u�r�y�zÇÓÕàì÷íìàÓÇ�z�z�z�z�y�m�`�T�O�G�F�G�T�Y�`�m�w�y���������}�y��Ѿž������ľʾ׾����	��!���	�������j�X�R�N�H�I�Q�\�s��������������������������������������������G�?��������"�.�6�;�@�G�O�V�[�Z�T�G�������������������������������������������j�b�n�����лܻ���������лû����ù��������ùϹչչϹùùùùùùùùùü����������ּ���!�,�6�;�<�:�0�!����ּ�ĳĭĦĦĦĪĳĳĳĿ������ĿĳĳĳĳĳĳĿĻĺĿ�����������'�!������������Ŀ��ż������������������������������������s�g�Z�N�C�J�N�^�g�s����������������������|���������������������������������
�	���������	���"�(�.�/�.�'�"�����������������������ʼͼּ׼ּܼռʼ����X�N�Z�Z�i�s���������������������s�f�XĿĴĶĿĿ����������������������ĿĿĿĿ��������(�)�5�8�=�5�)���������s�n�l�f�o�s���������������������������(�$���(�1�5�A�L�N�S�P�N�J�A�5�(�(�(�(�׾׾־оʾ��������������������ʾ׾׾׾������������������������������������������H�D�=�;�A�H�T�a�m�z�������}�z�q�a�Q�J�H��
������)�)�)�)�%� ����������}�z�m�m�e�m�z�������������������������ϹŹùùʹϹڹܹ������������ܹϺ����������!�'�-�!�������������{�q�n�i�n�q�{ŇŔŠŬŭŮŭţŠŔŇ�{�{�h�e�[�O�J�O�[�h�tāĆăā�t�h�h�h�h�h�h�)�%�$�)�6�9�B�H�B�6�)�)�)�)�)�)�)�)�)�)�g�[�[�T�[�_�g�t�t�~�{�t�g�g�g�g�g�g�g�g���������������������������������������������������
�����
�������������������#��������������
��.�=�E�U�Z�U�I�0�#������������������������ֺκɺúɺҺֺ�������������ֺֺֺֺr�Y�L�B�F�O�Y�r�~�������������������~�r�@�<�4�'��'�(�4�@�M�M�Q�M�C�@�@�@�@�@�@ƳưƧƚƋƁƀƁƎƚƧƳƽ������������Ƴ������������������������	�� � ������������������������������������������������6�4�+�.�6�C�O�\�h�p�s�h�g�\�O�C�6�6�6�6�Ŀ����������������ѿݿ��ܿۿ�߿ҿѿĻ����������}��������������������������������������������������������ǾʾϾ˾ʾ����������������$�0�<�4�0�(�$�"��������������ûɻлڻڻлͻû�����������������������������������������������<�2�/�,�)�)�-�/�<�H�U�`�`�X�U�T�L�H�<�<ÓÐÇÃÅÇÓàìòììáàÓÓÓÓÓÓ���������������������������������������������ÿú����������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� v M B ; $ 6 8 4 a , O . � H A ] B p   d Z E s 9 T 5 Y 7 h x 0 K 4 = K K B : 6 5 d 1 J 1 2  F q s X h 7 t @ � f / : Z c 3 9 M @ N F [ 4 r a Y ; t � D + B _ H T  �  �  }  �  �    1  �  3  ^  h  :  �  �  �  '  �  �  d  �    �  +    �  #     K  D    �  �  5  �  �  a    \  �  �  �  �    �  �    �  �  g  �  �  �  �    �    �  �  �  �  h  �  d  s  �  \  .  -  u  �  �  =  c    {  �  �  �  b  �  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  ?  �  �  �            �  �  �  �  �  n  H    �  �  r  C    *  $         �  �  �  �  �  r  J    �  �  8  �  �  f  �  D  F  C  @  5  *      �  �  �  �  d  &  �  �  q  C    �    C  V  ^  u  �  �  �  �  �  �    ]  3    �  �  ~  d  u  !                    
       �   �   �   �   �   �   �  �  �  �  �  �  �    z  x  v  t  r  p  q  z  �  �  �  �  �  z  �  �  �  �  �  �  �  w  d  G  &  �  �  �  @  �  \  �    .  "      �  �  �  �  �  �  |  [  :    �  �  �  �  a  ;    #  '  &  $  '  7  E  Y  m  r  c  D    �  �  x    m    �  �    +  K  f    �  �  �  �  �  �  �  u  ]    �    s  
  $  *  ,  1  6  :  >  =  1    �  �  �  3  �  �  3  �  g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  h  X  c  \  V  X  _  i  k  k  u  {  b  9    �  �  I    �  �  i  �  �  �  �  �  �  �  �  S  "  �  �    @  �  �  C  �  �  O  |  k  T  :    �  �  �  �  �  �  m  B    �  �      �  O             �  �  �  �  �  �  �  �  �  v  X  6     �   �  {  n  a  T  H  @  7  /  &        �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  t  j  `  W  N  D  ;  2  -  *  &  #    z  p  g  ]  S  C  3  $      �  �  �  �  �  �  �  y  `  G  >  .      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     �  �  �  �  �  �  �  �  �  �  �  i  d  ]  N  .  �  �  r  �  �  �  �  {  W  :  W  �  �  {  Z  #  �  �  u  F    �  �  �    �  �  �      *    �  �  e    �  n    �  7  �    b  R  @  ,    �  �  �  �  j  ;    �  �  =  �  �  6  �  �  �  �  �        ,  *      �  �  �  �  `  @  #    �  �  �  �  �    -  7  /    �  �  �  r  /  �  �  v  	  �    E     $  4  (    �  �  �  z  Y  9    �  �  j  "  �  U  �   �  �  �  �  �  �  �  �  �  }  t  l  d  \  V  V  V  V  U  U  U  �  �  �  �  �  r  ]  H  1      �  �  �  i  .  �  �  e    �  �  �  �  �  �  �  �  �  �    w  p  i  b  [  T  L  E  >          �  �  �  �  �  �  o  :  �  �  d    �    �   �  �  �  �  �  �  �  �  �  �  �  �  �  v  c  P  =  (    �  �  b  y  u  f  K  )    �  �  �  �  �  ^  #  �  P  �  �  `  �  �  �  �  �  �  ~  g  P  6    �  �  �  X  %  �  �  �  �  �  Z  =  !    �  �  �  �  �  �  �  v  ]  6    �  �  1  �  �  �  �  �  �  v  M  )  E  I  @  '    �  �  �  �  T  :  F  }  �  �  
    ,  7  5  #  �  �  �  X    �  S  �  =  �  �  }  �  �  �  �  ~  K    �  �  K  �  �  ?  �  p  
  �  R     �  �              	  �  �  �  �  b  9    �  �  �    �  /  ,  )  &  #        !        �  �  �  �  �  �  �  o  ]      �  �  �  �  �  �  �  �  {  Y  3  
  �  �  Z    �   v  R  Z  [  W  R  K  @  3  %      �  �  �  �  �  �  c  3  �  �  �  �  �  �  �  �  ~  q  ^  J  7  /  /  /  .  E  a  }  �    
    	    �  �  �  �  �  �  d  5  �  �  C  �    7  �  b  l  |  �  �  �  �  �  p  Z  <    �  �  �  W    �  �  �  \  W  Q  L  F  @  9  3  (    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  i  Z  K  <  +      �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  i  U  :    �  �  �  a    �  �  �  �  �  �  �  �  x  k  ^  O  A  4  *      A  f  �  c  i  o  u  z  ~      q  `  L  4      �  �  �  �  �  �  	r  	�  	�  	�  	p  	T  	'  �  �  �  X    �  Y  �  �  �  �  E   �  V  T  R  P  N  J  G  D  B  D  F  G  7       �  �  �  �  }  y  �  �  �  �  �  �  �  �  �  �  �  l  C    �  �  g  <      �  �  �  �  X  .    �  �  z  I    �  �  �  �  �  �  �  $      �  �  �  �  �  �  �  �  �  �  s  g  n  {  �  �  �  �  �  �  "  D  \  p    �  �  z  e  D  #     �  �    H    $  #  "          �  �  �  �  �  t  Z  ?  %    �  �  �  '      
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    /  9  2      �  �  �  b  E  )    �  �  �  N  �  x  �  �  �  �  �  �  x  l  a  U  I  =  1  %        �  �  �  a  �  �  �  �  �  {  r  i  ^  S  D  2      �  �  �  x  M  "  U  �  �  �  �  �  �  r  W  7    �  �  �  �  N  �        �  �  �  �  �  �  �  �  �  �  {  l  [  I  7  %     �   �   �  	  �  �  �  �  �  �  �  �  u  g  X  H  8  %  
  �  �  V    y  h  T  8    �  �  �  �  �  �  r  Z  6    �  R  �   �   .    �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  h  Y  J  ;    �  �  �  �  �  �  �  �  �  p  ]  D  (    �  �  w  )   �    �  �  �  �  �  �  {  `  @    �  �  �  o  &  �  �  <   �  �  �  �  �  z  Y  7    �  �  �  �  j  K  *    �  �  M  ,  b  S  D  6  !  
  �  �  �  �  �  {  n  [  >        �   �   �  +  &    
  �  �  �  �  �  =    �  �  �  �  �  �  �  �  O  �  �  �  �  �  �  �  �  �  �  �  �  {  q  c  T  E  7  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	1  	  	   �  �  �  j  =    �  �  K     �  L  �      �  8  
  
  
  
4      #  -  
�  
�  
P  	�  	h  �  ]  �  A  �  c    	  �  �  �  �  �  j  U  :    �  �  �  �  W  /    �  �  �  V  @  )      �  �  �  �  �  �  �  �  �  �  �  �  �    P  �  �  �  p  P  1    �  �  �  �  i  I  (    �  �  �  a  0  	  	  �  �  �  K  �  �  Q  �  �  A  �  |    �  D  �  J  �