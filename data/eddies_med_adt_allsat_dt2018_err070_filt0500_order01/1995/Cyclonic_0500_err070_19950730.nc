CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�������       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�`�   max       P�:       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��v�   max       <ě�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?^�Q�   max       @Ftz�G�     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @A��Q�     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q            �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�>@           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �z�   max       <e`B       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��S   max       B2ʉ       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|�   max       B2Ӆ       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��R   max       B~w       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@�   max       B��       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          d       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�`�   max       P��       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�䎊q�j   max       ?�ڹ�Y��       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��v�   max       <�j       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?^�Q�   max       @Ftz�G�     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @A������     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�k�           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A   max         A       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����A   max       ?���+     �  Zh      '                  	   #   "   \         7   	            
         6               0         	                           	      
      .            (            d               	   )   $            *                N�LwO�GzN���Ni�{N@�O
{�O��,N�5P�FPyP�:N͖~O,�`O�K	OO�iP/�O��P��No��OV�YN��=P<Q,N�feO��N�,ZOBȋP&XN�"O�MN��cO<uN1YM�`�O��jOJ��N��NR͗Os�N��O�v�N��YO�;P	��NuJtOWD	O�>O���O'@�O$#�N��+PeYM��N�0?NS�NQe�N�fO�]�O��:M�s�OiL6N<�
O�p�O�7O���N�O���OG�|<ě�<���<�C�<u%   ��`B�t��49X�D���D���D���T���T���T���e`B�e`B�u�u�u�u��t����㼛�㼛�㼣�
��1��j��j���ͼ�����������������`B�����������o�o�o�+�\)�t���P���'',1�<j�P�`�T���aG��ixսixսq���y�#�y�#�}󶽋C���O߽�O߽������v� )4*)�      �������������������������������������������������������)/5=5)"�����������������������������	��������EHIPT[acdca^ZTMHEAEE^r�������������zaXW^)5Niug]\i���gNB7-*-)����).& ���������%)---*)$)5:AAC@;5.)'.6BOhqyrjhkh[OB6)$'h\OH6..6?O\huy��vslh������������������������������_t��������������tg^_����������������������������������������������������������������������������������������������������)35BNX[bf[NB:5)��������������������y��������������|{wuy������������������������������������	
#/9<:1/#
	TTVamqwomhaYTTTTTTTT������������������������

����������hhuw���uhehhhhhhhhhh���������������� !#')+,)������������������������������������������������������������������������������������gw��������������zhcg������� ����������HINUgkntsnacdbUPFFFH��������������������6<=DHIONUWUH<6666666{����������������vu{$1Ibn{����{nV<0#$/<HUliigfbU</#��������������������26O[hqttnh[WONIBB642�����������������������������
!
���{x���������������������������


�������DHU[\WUHCADDDDDDDDDD<BNTTQNNMB?;<<<<<<<<oty����������tlioooo5BNQW[]UB5)$#0<FLPSb^UOF<0#IIRUUVVWVUMIGGIIIIII����������������������������������������6<Hay������zynaUE;66������������)364,)�����egpt����tgeeeeeeeeeerty������������{tnorit���������������zqi������ںٺ������� ������������
����������������������� �/�4�1�#��
�������������������������������¦¦ª¦�I�D�B�F�I�T�V�W�X�V�R�O�I�I�I�I�I�I�I�I�g�c�Z�U�X�Z�]�b�g�s�~�����������w�s�g�g�Z�N�I�L�Z�g�������������������������g�Z������������Ƹ����������������� �����������ֿĿ��������Ŀ����� ���(����m�h�s�z��������������������������z�m�л��л�������4�Y�r���������M����ܻпѿƿĿ��Ŀǿѿݿ������������ݿѿѿѿ������s�l�m�s���������������������������������|�t�t�������������ûɻӻػԻܻǻ����y�z�q�m�_�T�G�C�;�4�3�1�3�;�F�H�T�`�m�y�/����������������/�H�T�[�Y�M�>�4�3�/����+�(�4�M�f�������t�f�Z�A�4������ܾξ����������׾�	�"�.�=�<��	�����H�>�<�4�2�<�H�U�[�X�U�U�H�H�H�H�H�H�H�H�`�T�R�P�P�T�`�m�y�����������������y�m�`�e�]�Y�R�Y�e�i�r�~���������������~�r�e�e�I�F�0������ƳƚƎƆƈƚƳ������0�F�I�m�i�`�U�T�Q�R�T�`�m�n�y���~�y�n�m�m�m�m�������������������������������������������޿�������������������ĿĿǿѿݿ��������%�������ݿѿĺȺ��������ɺ���!�>�@�=�B�@�8�!����ֺȺ�����������!�)�!����������������#��#�(�)�.�<�H�U�a�c�d�a�[�U�K�H�<�/�#�"���
���"�/�3�;�F�;�/�,�"�"�"�"�"�"�����������������������(�.�/�)���
��������������
����
�
�
�
�
�
�
�
����������������������������������������ßÔÐÇ�p�d�h�n�zÇÝãí������ýùìß�[�O�B�=�=�M�[�h�t�yāčĘĎąā�{�t�h�[�u�s�h�_�h�s�u�wƁƆƁ�v�u�u�u�u�u�u�u�u�������������ʾʾʾȾž�������������������ŹųŭŠŚŕŠŭŹ���������������������U�T�P�U�\�b�n�w�{łŇőŇņ��{�n�b�U�Uā�h�O�H�B�E�Z�h�qāčěĒĘĜĩĦĚčāŔŒŔŔŘŠŤŭŹ��������żŹŵŭŠŔŔ�ܻٻл˻лܻ����������������ܼ� ����������4�@�Y�f�s�s�m�i�f�Y�'�����������������������������������������	�� �	����	��"�)�4�9�7�/�'�%�"��	���������p�d�]�Z�_�g�s�~�����������������ʾ������������ʾ׾������������׾ʻF�:�-�!��!�#�-�:�Q�_�l�r�|�z�x�l�_�S�F���}���~�������������������������������������������������������������������������$��)�%�������$�0�I�YǒǑǈ�V�=�$����'�,�4�@�M�M�M�@�;�4�'�����������������������)�,�1�0�)�����;�6�1�;�H�T�\�_�T�H�;�;�;�;�;�;�;�;�;�;�������������������������������������*�%�����*�6�C�I�O�O�O�L�C�6�*�*�*�*�a�\�]�Y�V�aÇîù����������óàÓ�z�n�a�����������������Ľܽ����ݽӽԽӽʽ����(�'������(�4�8�A�C�A�4�(�(�(�(�(�(�������~�t�g�~���������ɺֺں�޺ֺ˺ɺ��'���'�0�3�@�G�D�@�3�.�'�'�'�'�'�'�'�'�������������ù�����������Ϲ��������������!�.�:�E�A�:�.�,�&�!�����y�]�Q�`�x�����������ĽнܽڽнĽ������/�,�#���#�/�/�;�3�/�/�/�/�/�/�/�/�/�/��¿²¥¡²¿��������
��
����������������������������������������������� l 6 2 / � 9 W n _ q e * P  G C 9 E J , W f 3 � H N  O ? D B  y e M 1 J z ? > X :   O z j & W W O W � - 8 E * 8 _ � ` _ b O L K J k    �  ^  �  }  �  B  -  �  =  �  �  �  �  w  �  r      �  �  �  �  �  �  �  �  �  �  k  �  �  �  ?  �  �  +  �  �    �    ,  X  �  Q  �  _  �  �      a    ]  {  �  (  H  F     b  V  =  �  J  �  �<e`B����;�`B<t��o��t���㼣�
�<j�<j��񪼛�㼣�
��C���j��w�o�0 ż��ͽ+�ě���t���9X���ͼě��+��hs��/�L�ͽC��'���`B�aG��Y��t��C���P�#�
�]/�,1�t����w��w�P�`�Y����w�y�#�ixս]/�z�Y���o�q���}󶽇+�����ě���o�������S���ȴ9�� Ž�xս�
=Bx�B|B�MB�BpBռBA��SBPB!BAB<RB
�Bi�B1��BmB�B�QB!<�B*�B"�gB�B�B��B*	VB
��B"�/B,E�B��A�ѹBI�BK=B2ʉBtfB�gB�EB!�UB�|B�B ��Bl[B'x�B!��B0�B �B'�B��BM�B��Bt�B1B*�#B)B�*B�B
N�BcpB%�B'.B(UBB�B7ZB�vB=�B	��B
�-BC�B�LBC_B��B"B>)B��B�JA�|�BB�B<�BBBB@B1�B�B��B��B!<�B*�B"��BзB��B0B*-GB
��B#8�B,UpB��A�y{B?�BKB2ӅB��B��B�B!]�B��B��B �9B��B';�B!@aBGJB �B&�>B�[BA�BAuBmuBƤB*��B<�Bu�B�2B
��BW"B%�fB&��BAmBB�B?�B�B�/B	�rB
�pB
�@K�A��A��EA�AMB~wA�>�A�1B��A�w�A���@��A|�1A�!�@��+Af#�A���A;gAWA�g�AlC+?��B}�Aj;�As�A�	gA��@T@\WA�`�A��,A�qA�5'AL$A���Aہ�B��AN��A�/,A�s�Aܱ#A���@��j@�&A�o�A��oA���AS��@�~�A�hA�A�B`N@�A�AԴ�A��RA���B &NAʲ'A#��A7<�@�I?���>��RA�xA "KA�[A���A��@J?�A�u�A��CA�<B��A���A��KB�PA��A�X@���A|�A��v@�9�AgQ?A�{�A:��A[R"A��Ak}?��jB	t
Ai|�As��A�'�A�@U�j@aAĀA��A�mfA�w�AM�A��A�~�B��AN��A�{�A�k-A��KA�~�@���@�C�A�d�A�u�A��xAS�@��A�uA��HB
�#@��A��dA�u�A�t�A��zAɮA"JA9�@#�?�"�>@�A#A��A�x�A�g�A�!      '                  
   #   #   \         8   
                     6               1         	                           
            .            )            d               
   )   $            +            !                        %      /   1   I               '   '   -            1               '                     !                  #         '         +               3                  !   #            %      #                                          '                     -            !               !                     !                           !         +               %                                 %      #         Nw�xOV�N���Ni�{N@�O
{�O�X<N�5Of�VOk��O�oN͖~O��O` @OO�iO�J�O��P��No��Oi|N�� O�٤N�feO��N�,ZO*��O��eN�"N�bN��cN�PN1YM�`�O��;O�NN��NR͗Os�N�Op��N��YO�;O���NuJtN���O�>O��O��O$#�N��[O׬"M��N�0?NS�NQe�N�fO΁�O5 �M�s�OiL6N<�
O�p�O�7O���N�O�
�OG�|  �  �  �  _  	  �  5  c  #    �    �  �  m  �  �  l  �  T  <  H  �  �  �  6  =  1  #    �  �  �  �  �  W  W  �    �  �  q    2  �    7  G  �  _  �  -  h  �      �  �    �  \    K  �    �  �<�j;��
<�C�<u%   ��`B�T���49X������9X�D���T���e`B�����e`B���
����u�u���㼛��C����㼛�㼣�
��9X��P��j�+������h����������h�C��������+��P�o�o��P�\)�,1��P�49X�0 Ž'0 Ž��w�P�`�T���aG��ixսixս}󶽉7L�y�#�}󶽋C���O߽�O߽�����1��v� )2))      �������������������������������������������������������)/5=5)"����������������������������������������EHIPT[acdca^ZTMHEAEE�����������������z{�35BN[afb[VUUNB?;32/3���������������%)---*) %)58??B>85) )+46BOYaeb``[OB60+()h\OH6..6?O\huy��vslh�������������������������������_t��������������tg^_����������������������������������������������������������������������������������������������������)35BNX[bf[NB:5)��������������������w���������������}|xw�����������������������������������#//53/#TTVamqwomhaYTTTTTTTT������������������������

����������hhuw���uhehhhhhhhhhh������	�����������!$)))������������������������������������������������������������������������������������imu}�������������zni������� ����������HINUgkntsnacdbUPFFFH��������������������6<=DHIONUWUH<6666666��������������������$1Ibn{����{nV<0#)/HU]adcaYUH</# ��������������������26O[hqttnh[WONIBB642��������������������������
����������������������������������


�������DHU[\WUHCADDDDDDDDDD<BNTTQNNMB?;<<<<<<<<oty����������tlioooo)5BNQVZRB51)&#0<BIILMMI<0#IIRUUVVWVUMIGGIIIIII����������������������������������������6<Hay������zynaUE;66������������)364,)�����egpt����tgeeeeeeeeeesuz�������������tppsit���������������zqi�������ۺۺ�������������������������������������������
���"�(�#� ��
���������������������������������¦¦ª¦�I�D�B�F�I�T�V�W�X�V�R�O�I�I�I�I�I�I�I�I�g�c�Z�U�X�Z�]�b�g�s�~�����������w�s�g�g�g�Z�S�P�O�R�Z�g���������������������s�g������������Ƹ����������������� ����������ݿܿտҿԿݿ�������������������|�������������������������������������������'�4�@�a�o�p�M�@�'������ѿƿĿ��Ŀǿѿݿ������������ݿѿѿѿ����������s�o�o�s���������������������������������z�}�����������ûǻλ̻ʻû������y�z�q�m�_�T�G�C�;�4�3�1�3�;�F�H�T�`�m�y�	�������������	��/�H�Q�Q�M�D�2�/�"��	�(�����(�4�A�M�]�f�v�n�f�b�Z�M�A�4�(���ܾξ����������׾�	�"�.�=�<��	�����H�>�<�4�2�<�H�U�[�X�U�U�H�H�H�H�H�H�H�H�`�V�T�T�S�T�`�m�y���������������y�m�`�`�e�b�Y�e�r�~����������~�|�r�e�e�e�e�e�e�������������������$�0�6�>�@�<�2�$���m�i�`�U�T�Q�R�T�`�m�n�y���~�y�n�m�m�m�m�������������������������������������������޿�������������������ѿſȿѿݿ���������!�������ݿѺ��պͺú��źɺֺ���!�-�/�-�-�)�!��������������!�)�!����������������<�5�2�<�<�H�U�Z�^�U�T�H�<�<�<�<�<�<�<�<�"���
���"�/�3�;�F�;�/�,�"�"�"�"�"�"�������������������$�)�*�+�)�����
��������������
����
�
�
�
�
�
�
�
����������������������������������������ßÕÑÇ�r�e�n�zÇÙáìû������üùìß�[�O�B�B�B�O�Q�[�h�t�āćĉā�}�t�s�h�[�u�s�h�_�h�s�u�wƁƆƁ�v�u�u�u�u�u�u�u�u�������������ʾʾʾȾž�������������������ŹųŭŠŚŕŠŭŹ���������������������b�Y�U�U�U�_�b�n�{ŇŃ�{�z�n�b�b�b�b�b�bčă�t�h�V�M�K�O�b�h�tāćČĔĚġĠĚčŔŒŔŔŘŠŤŭŹ��������żŹŵŭŠŔŔ�ܻٻл˻лܻ����������������ܼ��������	���@�Y�f�o�o�k�j�f�Y�'����������������������������������������������
�	�	�	��"�#�-�/�2�/�/�#�"�����������p�d�]�Z�_�g�s�~�����������������ʾ����������ʾ׾���������������׾ʻF�:�-�#�)�-�:�F�S�_�l�p�x�z�x�w�l�_�S�F���}���~�������������������������������������������������������������������������I�4�'�����$�0�=�I�V�d�|��x�o�b�V�I����'�,�4�@�M�M�M�@�;�4�'�����������������������)�,�1�0�)�����;�6�1�;�H�T�\�_�T�H�;�;�;�;�;�;�;�;�;�;�������������������������������������*�%�����*�6�C�I�O�O�O�L�C�6�*�*�*�*�^�_�_�\�a�nÇù������������íàÓ�z�n�^�����������������������ĽϽ̽нνĽ������(�'������(�4�8�A�C�A�4�(�(�(�(�(�(�������~�t�g�~���������ɺֺں�޺ֺ˺ɺ��'���'�0�3�@�G�D�@�3�.�'�'�'�'�'�'�'�'�������������ù�����������Ϲ��������������!�.�:�E�A�:�.�,�&�!�����y�]�Q�`�x�����������ĽнܽڽнĽ������/�,�#���#�/�/�;�3�/�/�/�/�/�/�/�/�/�/��¿²§¤²¿������������
������������������������������������������������ j / 2 / � 9 P n 4 ] E * P  G @ + E J  ] C 3 � H J  O . D =  y j F 1 J z 5 D X :  O i j   L W L \ � - 8 E * 7  � ` _ b O L K G k    �  N  �  }  �  B  �  �  �    Q  �  .  �  �  �  *    �  /  �  �  �  �  �  �  �  �  �  �  "  �  ?  �  d  +  �  �  �  �    ,    �    �    `  �  �  d  a    ]  {  �  �  z  F     b  V  =  �  J  .  �  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  g  U  B    �    ]  �  �  �  �  �  �  �  �  n  >  �  �  :  �  K  
  �    �  �  �  �  �  �  �  ~  n  Z  E  0      �  �  �  �  �  �  _  \  Y  V  S  N  J  D  ;  3  (      �  �  �  �  �  i  >  	    �  �  �  �  �  �  �  �  �  �  �  �  �  v  e  S  A  0  �  �  �  �  }  k  U  ?  $  	  �  �  �  �  c  =    �  �  �  $  -  1  0      �  �  �  �  f  D    �  �  �  �  +  �   �  c  Y  O  C  7  #    �  �  �  �  _  ,  �  �  7  �  �  `    �  4  �  �  �         "      �  �  �  N  �  �    `  u  �  �  �  �  �  �  �  �  �  �  �  �  �  O    �  q  �  �    C  [  K  ,  *  .  P  {  �  k  D  
  �  [  �  �  .  �  )  0          	    �  �  �  �  �  �  �  �  i  L  .     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  d  2  t  �  �  �  �  �  �  �  s  K    �  �  #  �  �  �  �  �  m  g  b  Q  >  *      �  �  �  �  �  u  Z  =     �   �   �    S  u  �  �  �  �  z  e  I  >  Y  Q  1  �  �  q  -  �    �  �  �  �  �  �  �  �  �  �  �  v  N    �  �  �  a  9   �  l  a  c  T  G  =  2  #    �  �  v  8  �  �  �  T  (      �  �  �  �  �  �  �  �  �  y  O     �  �  �  I  	  �  �  F    (  ;  L  S  S  H  8  #    �  �  �  �  g  /  �  �  0   �  2  5  8  <  ;  9  8  7  7  7  0          �  �  �  �  �  t  J  i  �  �  �  2  F  8    �  �  f  #  �  i  �  �  �    p  �  �  �  �  �  �  �  �  u  f  W  H  8  *         �   �   �  �  �  �  �  �  �  �  �  �  �  p  W  ?  /  1  3  1      �  �  �  �  �  �  �  �  �  x  o  h  `  Y  R  J  :  (       �  (  1  4  1  (      �  �  �  �  �  |  N    �  }  3   �   �  �  �    )  4  <  9  &  	  �  �  �  b  +  �  �  v  5  �  5  1  #      �  �  �  �  �  �  �  s  ^  J  6     
   �   �   �  �  �  �  �      "           �  �  b    �    �    �      �  �  �  �  �  �  �  �  �  i  O  /    �  �  �  �  Z  L  Z  g  w  �  �  �  {  x  x  {  l  R  1    �  z  #  �  .  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  j  _  T  I  >  3  (         �   �   �   �   �  �  �  �  �  �  �  m  Y  e  �  �  x  ]  ;    �  7  �  L  �  �  �  �  �  �  �  �  �    V  +  �  �  �  v  G  �  F  H  1  W  A  *    �  �  �  �  �  �  e  H  ,    �  �  �  f  1   �  W  I  ;  -        �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  {  i  W  D  0      �  �  �  �  �  u  ^  I  5     �    
        �  �  �  �  �  �  �  �  �  �  [  7  #    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  G    �  �  �  ~  v  l  a  N  9  !  	  �  �  �  �  {  Z  5    �  �  �  q  n  l  i  g  c  [  S  K  D  >  9  5  1  -  *  )  (  &  %           �  �  �  �  �  c  7    �  �  ^    �  K  �  �  2  *  !        �  �  �  �  �  �  �  �  �  �  Z  .     �  �  �  �  �  �  �  �  �  �  �  �  �  \  /  �  �  �  s  5  �      �  �  �  �  �  �  w  U  2    �  �  �  �  r  C     o  �    -  6  .    �  �  �  D    �  k    �  2  �  �  Q  '  !  =  F  ?  3      �  �  �  h  6    �  �  L  %  "  �  �  �  �  �  �  �  �  �  {  e  G  #  �  �  �  �  �  �  �  �  �  Y  ]  ^  ^  [  X  S  P  M  F  9  )    �  �  �  �  �  �  �        
�  0  }  �  }  w  \  -  
�  
e  	�  	  4  h  ?  N    -  (  #          	    �  �  �  �  �  �  �  �  w  f  T  h  P  :  '    �  �  �  �  �  b  =    �  �  v  =  �  �  �  �  �  �  �  �  �  �  �  |  q  g  ^  V  M  D  8  *           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  z  t  v  �  �  �  �  x  d  O  9  #    �  �  �  �  �  �  �  �  �    U  +  �  �  m    �  j     �  �  4  �  �  �  i  �  �  �  �  o  L     �  �  v  0  �  �  7  �  9  J           %  *  /  1  0  /  .  .  -  -  ,  ,  ,  ,  ,  ,  �  �  �  �  d  A  !  	  �  �  ~  ;  �  �  `  )  �  �  x  a  \  Y  W  S  N  I  ;  %    �  �  e  /  �  �  �  I     �   �    �  �  �  �  �  �  n  N  ,    �  �  i    �  J  �  �  �  K  4      �  �  �  �  �  �  �  �  �  �  �  u  ^  ;    �  �  �  �    e  G  *      �  �  �  Q     �  �  �  $  �         �  �  �  �  �  �  �  p  M    �  �  �  Q     �   �   �  x  �  n  M  %  �  �    7  �  �  H  �  �  L  �  �  0  h    �  �  �  r  [  B  )          �  �  �  �  a  0  �  �  &