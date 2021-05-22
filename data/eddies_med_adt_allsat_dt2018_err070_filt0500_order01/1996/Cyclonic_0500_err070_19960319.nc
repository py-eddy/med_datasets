CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�z�G�{        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��J   max       P�y        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <e`B        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>^�Q�   max       @F��\(��     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�X    max       @vl�\)     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @��@            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �n�   max       <D��        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�;   max       B4st        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4��        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�q�   max       C�/q        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >UPc   max       C�*        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          P        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��J   max       P��        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u&   max       ?��MjP        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��
=   max       <e`B        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>fffffg   max       @F��\(��     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�X    max       @vlQ��     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�x�            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u%F   max       ?����l�        W�      #            	            
               
      	   P            	         %            	                        G                              %      '   !   
            
         
                  1N��O��O���OvbwN��NQ�bOvN�^�O&��O* N<G�O6�O���N��eN�0�N8F�N��sP�yO̍�N��N���N��iO<G�O��{P;N!87Oi�gO�GOO��NR��N���O�)�O9��N�mO��N��RO��OQ�!Ov�N/��N�Ok��N�N��N���N���O�N<OF��P�P��M��JO i)ONr�O��NI�|N4p�Ov֟N���Nq6O&zNH�'Nfa�N�Os�<e`B<T��<o;ě�;ě�;��
;o%   ��o��`B�o�t��#�
�49X�49X�D���D���D���T���u��o��t����㼣�
���
��j�ě�������������������/��`B��h�������o�C��C��C��\)�t��t���P�#�
�#�
�#�
�49X�49X�D���H�9�P�`�P�`�e`B��%��o��O߽�\)���㽟�w��{����������������������������������������������hlw�������������tmfh���   �������<BINQRVTNIB?95<<<<<<W[hkt�tqh[YUWWWWWWWW���������������������������������������������		��������DHIMRU[annnpnmibaUHDENR[ac[NIDEEEEEEEEEE
#/29<=<7/##
��������������������������������������������������������ABOY[hkh[OGBAAAAAAAA��
#&.#
��������7<b{����6EC����{bI7")6BOTUTUQQNB6',056>CHHHFC=64*����������������������������������������FHTahmmmfba\TPLHDBBFgt������������tg]_bg��������������������"#/<>C</%#"""""""""")585:BD>5)��������������������gitw�����������ztg^g#/<AB=<9/#!TUVajnz|zuna\UOHTTTT���������������������������������������������������������������

�������#/5<UUUKH<4/# 4>==N[mg_^gt|{p[NB94`dgmtz}�������zmlda`��������������������~�����������~~~~~~~~fhlt���|thgeffffffffNR[t����������tgeUNNV[hmih_[ZVVVVVVVVVVV��������������������@BOS[\c[[ODB?=@@@@@@������������������������5)%	������6BOT[ht{��}yth[OMB:6��������������������	'&<Ugx{qjbI0
	��������������������������������������������������������������������	�����������������������������X\gt����������tkg[X�������������������������������������������� �����������������������������������
 !
	#'5AE?8/&!�������������ʾ;ξʾ������������������������������������
�/�H�U�a�[�H�>�/�����ݿѿĿ������Ŀݿ����(�5�8�)�+�)������������������	��$�=�K�S�P�I�=�0�$�������}�y�r�y�������������������������������e�]�b�e�o�r�x�~�����~�r�e�e�e�e�e�e�e�eàØÒÑÎÓÖàèìùþ��������ûùìà�;�7�1�/�'�/�;�H�T�a�m�s�m�m�a�Z�T�H�;�;�	������������������	��%�-�/�3�/�"��	�g�]�Z�A�(�"�����(�5�A�N�Z�g�g�m�q�g�Z�W�Z�]�f�s�{�x�s�f�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z��������������������������������������������������������������������������������ŹŭŢŠŔŋŇŅŇŇŔŠŢŭŭŹŹ����Ź��|�s�k�f�c�f�p�s�v�������������������a�\�Y�a�c�f�n�q�z�u�n�f�a�a�a�a�a�a�a�a�������������������żƼ������������������ʼ����������������)�5�I�U�����ڼؼʿT�G�;�����پ����"�;�T�`�m�v�w�p�m�T�"������׾Ծʾľʾ׾����	���"�$�"������������������
���������������f�Z�Y�O�M�L�M�Y�f�k�r�z�|�r�f�f�f�f�f�f�������(�5�N�Z�`�g�v�g�Z�S�A�5�(��l�a�Z�W�Y�a�m�z���������������������z�l�`�T�B�<�7�7�?�G�T�X�k�������ÿ����y�m�`�H�F�<�A�H�U�X�Z�U�N�H�H�H�H�H�H�H�H�H�H��������������������������������������������������������
�<�N�U�]�_�U�I�<�#���������������� ���0�5�4�)�����6�,�1�4�6�B�O�R�O�I�C�B�6�6�6�6�6�6�6�6�a�`�a�d�m�q�y�z�|����������z�m�a�a�a�a�;�9�;�A�@�J�T�a�h�s�w���������m�a�T�H�;�������{���������������½ɽȽĽ���������������������������������������������������������������������������������������������������������������������������������B�������þ����������"�B�O�[�e�g�f�[�B�"��	�������������	��"�/�;�E�H�H�;�/�"����������#�(�5�H�N�H�A�5�(�����������������������������������������������������'�+�'�'���������������������	����"�'�1�,�$�"��	���ù����ùϹֹܹ�ܹϹùùùùùùùùùþA�9�4�4�4�A�M�Q�Z�^�Z�M�A�A�A�A�A�A�A�A�l�h�g�l�x�x���������������x�l�l�l�l�l�l�ܻӻлû����ûлܻ�����ܻܻܻܻܻ��s�g�X�M�Q�\�g�q�����������������������s���������~�}�|���������������������������ֺ˺Ǻĺ����ֺ��!�:�F�F�<�)�!����ֻл����l�_�F�:�7�M�l���м��%����ݻ������������������������������������������������
�����*�0�0�1�,�*�)�����x�m�`�U�S�H�F�@�A�F�S�_�l�x������������޺ֺԺغ�����!�-�<�=�*�!������������)�6�B�M�B�6�)���������������������������������¦²´º¿¿¹²®§������$�%�0�=�>�C�I�K�I�=�7�0�+�$������������������
��
�����������������`�W�S�M�J�L�S�`�l�y�������������|�l�c�`�3�/�3�8�@�K�L�S�O�Y�Z�Y�L�@�3�3�3�3�3�3��y�{���������������������������E*E!E*E+E7ECEPE\EiE_E\EPECE7E*E*E*E*E*E*EuEiEuExE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eu ? I C E C G 8 H \ i H I 9 v b g C u h � / B [ * R > ; Q T h [ [  ( . b s < M z F @ F R 9 G 0 : 8 C r k w F � K L I [ 5 O t _ G  %  X  �    �  }  F  8  �  �  c  H  <  7  1  �  �  �    o  �  �  �  �  �  N  �  ^  �  �  �  �    �  c  S  �  �  &  �  =  �  !  2  �  �  W  �  b  y  ]  �      �  [      �  y  |  �  �  <D����9X����e`B�o��o��o�D���u��C��D����h�C���t���1���㼣�
��^5�+���
���
��/���L�ͽe`B��h���ixսC������0 ŽD���+�@���P���`�<j�H�9���'}�0 Ž#�
�@��aG����-�y�#�����ixսq�����P�}󶽅���7L��1���w����\����\���n�B4stBZ!B
��B�rBB��B(�BQ\A�;B�_Ba�B��B�QB�%B��B�B$y�B,\$B�qB/� B B ��A���B
ĞB+ �BSB��B=B
�rB"�B%.B��B#pBFB�cBb�B��A�Y�B�
BFgB&B	��Bs�B"�MB�3Bj�B��BehB��B&�VBslB�B!$�B"��B.�B@�B
t�B�B�RB��B�bB�	B��B��B4��B6kB
>&B�AB��BB�BĦBYlA��BHDBFOB��B��B:QB��B �B$X�B- �B��B0;�B�B �zA��XB
��B*¢BE�BA�B?�BO'BC8BQ�B�B"�3B�B��B� B��A��!B�B��B6�B	�JB~�B"ذB�B�kB��BB_B�B&ʺB��B�PB �B"�tBD�B?rB
>�B�XB?OB� B=�B��B8~B�-AOo`A�i�A�R�B	��Aq?�EA�$A�U�A��lA���AA5�A���A�FmA�˴AFe�A��
@���A��A`��AW�'A�.�@��A���A�2�Aj؆A�҄A�mqA�|�A�~�A�RA��A��TA"�BAQA�^�A�jRAמA��fA���A�-�?�|AAZ��>�q�A<d:@��@���A���A���@UpR@���A��TA�\@�ޮ@\��A֒�A/�*A���B
1bA�.AV?�g�@�?DC��fC�/qAO�A��HA�UB
7�Ap��?��dA�yEA�YDA���A��=AAhA���A�}�A�mAE�&A�~@���A�NAa�*AW�vA�+o@�K�A�~�A�|aAjD6A�?A���A�|�A�]�A�h&A�T�A�v:A!��B�gA���A�w:A���A��sA�~>A�s2?�u{A[ >UPcA<!�@��@��_A��MA���@[C�@�:�AωjA�i�@��e@Y7A���A0��A�}�B	��A燙A�K?���@���C���C�*      $         	   
                                 	   P            
         %             	                        H                              &      '   !   
            
         
                  2      #   !                                             M   #                  +         !            #               )                              !      %   ?            !                                                                                    ;   #                           !                                                               %   ;            !                              N��O���O;AO#n�N��Ni�OvN��O�jO* N<G�N�G�OS��N��eN�0�N8F�N��sPs�,O���N��N���N��iN���O��{O^��N!87Oi�gO�GOO��NR��NS� O��DO9��N�mO��N��ROL�OQ�!OU��N/��N�O$bN�N��N���N�d�N�|�OF��P�P��M��JO i)OyoO��NI�|N4p�Ov֟N���Nq6O&zNH�'Nfa�N�OI�  &  �  �  ~    G  �  �    M    �  9    U  #  �  �  �  �    �  �  "  J  �  0  �  �  '    Q  �  �  �  �  �  *    �    H  �  �  �  =  \  f  H  �  �    w  �  @  �  �     &  (  �  G  	  	<e`B;o�o:�o;ě�;D��;o�o���
��`B�o�#�
�u�49X�49X�D���D���ě��e`B�u��o��t����ͼ��
�\)��j�ě���������������/��`B��`B��h�����aG��o�\)�C��C��'t��t���P�,1�ixս#�
�49X�8Q�D���H�9�aG��P�`�e`B��%��o��O߽�\)���㽟�w��{�����
=����������������������������������������qtz�����������tsomnq��	������<BINQRVTNIB?95<<<<<<Y[hhslh[[VYYYYYYYYYY�����������������������������������������������		�������DHIMRU[annnpnmibaUHDENR[ac[NIDEEEEEEEEEE
#/19<<<6/%#
 �����

�������������������������������������������������ABOY[hkh[OGBAAAAAAAA��
#&.#
��������r����&50������wpr !#*6BOSTSTPPLB6',056>CHHHFC=64*����������������������������������������GHRTadiifbaTNHHFGGGGgt������������tg]_bg��������������������"#/<>C</%#"""""""""")585:BD>5)��������������������gitw�����������ztg^g#/<AB=<9/#!LUahnwnna_URLLLLLLLL���������������������������������������������������������������

�������#/5<UUUKH<4/# BN[^gjnrusgb[NIB@>?B`dgmtz}�������zmlda`��������������������~�����������~~~~~~~~fhlt���|thgeffffffffS[\gt{��������tog[USV[hmih_[ZVVVVVVVVVVV��������������������@BOS[\c[[ODB?=@@@@@@�������������������������������6BOT[ht{��}yth[OMB:6��������������������
)-<UevzphbI0
��������������������������������������������������������������������	�����������������������������X\gt����������tkg[X�������������������������������������������� �����������������������������������
 !
	#,/8<<<<:1/#�������������ʾ;ξʾ������������������������������������
��#�/�<�B�;�1�#�����ݿؿѿʿȿѿۿݿ����������������	�������$�0�=�C�I�L�K�I�=�5�0�$�����}�y�r�y�������������������������������e�`�e�e�r�~����~�r�e�e�e�e�e�e�e�e�e�eàØÒÑÎÓÖàèìùþ��������ûùìà�H�<�;�6�;�;�H�T�a�m�q�m�j�a�W�T�H�H�H�H��
�	���������������	���"�$�*�,�"���g�]�Z�A�(�"�����(�5�A�N�Z�g�g�m�q�g�Z�W�Z�]�f�s�{�x�s�f�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z��������������������������������������������������������������������������������ŹŭŢŠŔŋŇŅŇŇŔŠŢŭŭŹŹ����Ź��|�s�k�f�c�f�p�s�v�������������������a�\�Y�a�c�f�n�q�z�u�n�f�a�a�a�a�a�a�a�a�������������������żƼ����������������������������Լ���!�5�;�I�:�!�����ּʼ��T�G�;�"�����۾����"�;�T�`�m�u�v�m�T�"������׾Ծʾľʾ׾����	���"�$�"������������������
���������������f�Z�Y�O�M�L�M�Y�f�k�r�z�|�r�f�f�f�f�f�f�(�'�����(�5�A�N�Q�X�N�B�A�5�(�(�(�(�l�a�Z�W�Y�a�m�z���������������������z�l�m�`�T�Q�I�E�D�G�T�`�m�y�������������y�m�H�F�<�A�H�U�X�Z�U�N�H�H�H�H�H�H�H�H�H�H��������������������������������������������������������
�<�N�U�]�_�U�I�<�#���������������� ���0�5�4�)�����6�,�1�4�6�B�O�R�O�I�C�B�6�6�6�6�6�6�6�6�m�d�g�m�t�z���������|�z�m�m�m�m�m�m�m�m�;�9�<�>�B�A�T�a�g�r�v�~�����z�m�a�T�H�;�������{���������������½ɽȽĽ���������������������������������������������������������������������������������������������������������������������������������)������)�6�B�O�Y�[�^�^�[�S�O�B�6�)�"��	�������������	��"�/�;�E�H�H�;�/�"� �����������&�(�5�E�L�G�A�5�(��� �����������������������������������������������'�+�'�'����������������������	���"�&�%�"���	���ù����ùϹֹܹ�ܹϹùùùùùùùùùþA�9�4�4�4�A�M�Q�Z�^�Z�M�A�A�A�A�A�A�A�A�l�h�g�l�x�x���������������x�l�l�l�l�l�l�ܻܻлû����ûлܻ����ݻܻܻܻܻܻ��g�a�`�g�m�s�������������������s�g�g�g�g���������~�}�|���������������������������ֺ˺Ǻĺ����ֺ��!�:�F�F�<�)�!����ֻл����l�_�F�:�8�O�l���м��#����ܻ������������������������������������������������
�����*�0�0�1�,�*�)���x�q�l�b�_�Y�S�[�_�l�o�x�}�������������x��޺ֺԺغ�����!�-�<�=�*�!������������)�6�B�M�B�6�)���������������������������������¦²´º¿¿¹²®§������$�%�0�=�>�C�I�K�I�=�7�0�+�$������������������
��
�����������������`�W�S�M�J�L�S�`�l�y�������������|�l�c�`�3�/�3�8�@�K�L�S�O�Y�Z�Y�L�@�3�3�3�3�3�3��y�{���������������������������E*E!E*E+E7ECEPE\EiE_E\EPECE7E*E*E*E*E*E*E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� ? E 7 A C 8 8 B R i H I ? v b g C _ j � / B = * 5 > ; Q T h ] ]  ( . b . < 6 z F 2 F R 9 6 0 : 8 D r k i F � K L I [ 5 O t _ 4  %  8  �  t  �  *  F  �  3  �  c  7  �  7  1  �  �  �  �  o  �  �  �  �  �  N  �  ^  �  �  �  r    �  c  S  �  �  �  �  =  P  !  2  �  �    �  b  c  ]  �  F    �  [      �  y  |  �  �  B  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  &  !          
        �   �   �   �   �   �   �   �   �   �   �    �  �  �  �  �  �  �  �  �  ]    �  �  ,  �  r    �  �  �  �    U  h  v    �  v  X  3    �  �  R    �  �  O    [  l  v  z  }  }  x  j  U  :    �  �  �  �  ^  2    �  r        	  	        �  �  �  �  �  �  �  y  _  C  %    D  @  <  @  F  A  8  .  #        �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  G    �  �  z  ^  S    �  W  �  Z   �  �  �  �  �  �  �  �  �  �  �  ~  r  e  X  M  C  6    �  �  �  �    
      �  �  �  �  �  �  �  �  Z    �  �  =   �  M  A  4  '      �  �  �  �  �  �  �  p  Q  1  �  �  Q   �           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  >    �  �  �  I     �  I  �  �   �    *  2  7  9  4  (      �  �  �  r  Q  3    �  �  �  �       �  �  �  �  �  �  �  �  �  �  �  {  W     �  �  C   �  U  I  =  4  -  *  &      �  �  �  �  �  h  G  %    �  �  #  #  #  !        �  �  �  �  �  �  �  �  u  a  K  4    �  �  �  �  �  �  �  �  �  �  �  g  N  )    �  �  �  x  \  >  D  �  �  �  �  �  J    �  �  �  �  �  6  �  A  �  �  �  �  �  �  �  �  �  �  �  r  O  *  	  �  �  �  a  %  �  �  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  Y  6     �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  k  �  �  �  �  �  �  �  �  �  w  ]  :    �  �  �  �  �  �  +  �  �  �  �  �  �  �  �  �  �  �  �  {  ^  @    �  �  ~  �  "        �  �  �  �  �  �  �  �  �  \  8    �  �  b  G  y  �  �       7  C  H  I  H  :       �  �  q    �  �  %  �  �  �  �  �  �  �  �  �  �  �  ~  {  x  t  q  q  �  �  �  0      �  �  �  �  �  �  �  �    b  >       7  h  �  �  �  z  a  >      �  �  �  �  �  ^  :    �  �    |  �   �  �  �  �  �  �  �  �  �  �  �  �  {  s  h  ]  P  C  6  (    '    �  �  �  �  �  �  �  n  U  8    �  �  �  �  �  �  �                    	           �  �  �  �  �  �  0  K  O  L  K  J  B  8  .       �  �  �  �  \  �  �  /   �  �  �  �  �  r  _  J  3    �  �  �  ~  L    �  �  S      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  j  \  N  @  �  �  �  t  b  O  :  %    �  �  �  �  �  K    �  �  h  �  �  �  �  �  �  �  �  �  �  u  h  Z  H  7  &      �  �  �  	  	�  
P  
�  b  �  �  �  m  ,  
�  
Z  	�  	L  �    	  �  B  �  *  !    �  �  �  �  �  f  I  ,    �  �  �  y  N    �  �  �      �  �  �  �  �  �  �  �  �  s  R  +    �  �  �  �  �  �  �  {  l  ]  J  8  %    �  �  �  �  �  �  �  �  �  �    $  4  @  A  B  B  ?  =  :  7  4  /  +  &  !          �    +  <  F  D  1    �  �  �  �  p  C    �    p  �    �  �  |  w  q  j  b  [  S  J  @  7  .  $      	        �  �  �  �  �  �  �  �  �    t  g  Z  M  @  6  .  &      �  �  �  �  �  �  �  �  �  �  �  m  T  8    �  �  G  �  w  $  1  9  6  &    �  �  �  �  �  u  R  )  �  �  _    �  8  �        &  3  C  P  X  \  R  ;    �  �  g     �  0  \  f  ^  Q  ?  -      �  �  �  �  z  U  *  �  �  �  v  S  %  H  5    �  �  �  �  |  _  A    �  �  O  �  h  �  �  f  �  �  }  k  U  B  3  0  2  /  !    �  �  �  �  V    �  Y   �  �  �  �  �  �  �  �  �  �  �  p  S  *    �  �  �  Y  -      b  D     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  &  v  s  n  h  ]  L  /    �  �  i  0  �  �  U  �  ~  �  �  �  v  `  E  5  '                        "  @  #    �  �  �  �  j  J  '    �  �  �  f  =    �  �  w  �  �  �  �  �  �  �  �  �  �  �  |  o  c  V  7    �  �  �  �  �  �  �  �  �  a  @       �  �  �  �  S  �  �  #  +   �     �  �  �  �  �  s  J    �  �  �  t  J    �  �  �  T    &  #  !        
     �  �  �  �  �  �  �  �  x  d  Q  >  (    �  �  �  �  �  X  4    �  �  �  \  #  �  �  �  �  �  �  �  �  �  �  x  b  L  5      �  �  �  �  �  |  f  O  9  G  0      �  �  �  �  �  �  �    f  D    �  �  �  X  "  	  �  �  �  �  �  t  V  8    �  �  �  �  �  h  /  �  	  A    �  �  	
  	  	  �  �  �  �  |  L    �  �  ?  �  U  �  �