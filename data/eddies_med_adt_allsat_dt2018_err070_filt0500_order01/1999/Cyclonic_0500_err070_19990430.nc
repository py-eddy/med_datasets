CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��E���       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Ns�   max       P8�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���w   max       <�t�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>u\(�   max       @F�z�G�     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����P    max       @vUG�z�     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @Q`           �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�            5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��w   max       <e`B       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�7�   max       B4�       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�JK   max       B5 +       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�u�   max       C�x�       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�ښ   max       C���       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          u       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Ns�   max       P8�       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����   max       ?�ᰉ�'S       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��G�   max       <�t�       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>u\(�   max       @F�z�G�     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vU�Q�     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P�           �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @���           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F`   max         F`       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�䎊q�   max       ?�ᰉ�'S     �  Y|         
   	   
      8      &   	   	         L                                           
   
                  &      a         C            u   .               (                     )                           #N��SN��O�nN���O�NlhSO}҆N�W�O��N�I�O��Op�O���P��O��-NRg�OBĔN��sO�wN���O
�6O{��O���O_�O^:�N/�]NU��OM�Ng�/N>�O$��O;��Ol�OT�O�l�NW9�P �&O�$O�d�P�fO�4N�.�O��KP$��P7xP-��N$	�NP��O{��O�P�N9��N�Y,O���N�b3Ns�NV��P8�N��N"I�N���N���O�U�O*-�O��%O�oO@w<�t�<�o<D��;ě���o�o���
�ě���`B�o�e`B�e`B��o��C���t���t����㼣�
���
��j��j��j��j���ͼ��ͼ���������/��h��h��h�����o�o�o�+�C��C��\)��w�#�
�#�
�#�
�,1�0 Ž0 Ž0 Ž0 Ž49X�49X�49X�8Q�8Q�8Q�D���L�ͽL�ͽT���aG��aG��u�u�u�y�#���wFO[\hilhe`[OLGEDFFFF""/46//" """"""""""#/6<@HRUVUULH</#)*32)'ou��������������trqo��������������������%&'#����������������������������������������������)56??6)DHJUUacmnsuynaULHADD��������������������6BOhmt�����t[B<32+,6������������������������������������������������������������$).6BEOQY\dda[OB6)"$���		��������������������������zz���������zyvvutsrz������������X[gt������������tdZX�������� ������������������������9<GIU_lmihdZUI<74349-/;@@?<;:0//--------��������������������������������������������{uty{���������������������������������������������������������	#0<PUZZWVI<0#
	QTVamy����zyma\TROOQ����������������������������������������7AUanz�������znUH?77��������������������[ht����������th^[PS[��������������������������������������<BDN[`dab^\[YNIB@;<<������������������������)��������5=N[t����������x[B95N[gt����������thNHINX[]gsmig[XXXXXXXXXXX���� ������������
#05>A<:40#
	N[got��������tgZPIIN���������������������#%$ ������������������������������������������������������������������������������������|����������������zx|�������������������������������������������������������EHUYaklaUHB=EEEEEEEE���� $#���������������������������� 
#.7<<:0#
�����)5;CIKB,	 �����
#/41/)$
�����e�a�e�e�p�r�~���������������~�r�e�e�e�e�#�"��#�0�<�=�=�<�0�#�#�#�#�#�#�#�#�#�#�M�C�5�4�4�4�5�A�I�M�Z�Z�f�q�h�j�j�f�Z�M�ֺѺɺĺɺɺֺ�������ݺֺֺֺֺֺ��"������������	�����/�;�<�;�/�"�f�_�b�f�g�s�����������s�k�f�f�f�f�f�f��ĿĴĳīĦĳ��������������������������������������������������������	�����������
��#�.�/�5�6�/�#�#��	�	�	�#���#�#�/�<�C�@�<�:�/�#�#�#�#�#�#�#�#�Z�P�N�A�A�?�A�N�P�Z�g�s�x�y�v�s�q�g�Z�Z���������������������¾ʾξԾԾʾ���������	����	���"�;�G�`�|�z�q�m�T�G�;�.��!���+�=�N�g�����������������s�Z�N�5�!�g�N�F�=�9�M�Z�f�s�������������������s�g���������������������������������������׾������}�g�f�Y�f�s���������������������������������	���"�+�"��	�����������������������������ʼּ����޼ۼּмʼ�ĦĦĲĳĿ������������������������ĿĳĦ�����׾ʾþ����������ʾ���������������޾־ؾҾҾ׾���	���%� ��	����ݾ׾վʾžƾҾ����"�-�0�.��	���������������������*�<�L�S�O�C�6�*�������������(�4�A�M�O�N�@�4�(����������������������������������������������z�y�y�y�������������������������������g�f�g�p�{�}�������������������������s�g����������������������ìêëìù��������ùìììììììììì���������������������������	������ݽнĽ������������Ľн۽����������ݽ��������~��������������ͽͽŽ���������������ƿ�������������
���������������������ƿƷ���������������� �����徾�������ʾ׾����׾ʾ����������������ù����������ùϹܹ���$�%�������Ϲ����������������������������&�"��������������������Ľнݽ���������ݽн����������!�1�Y�~�������~�m�L�3�����������������ʼּ���ݼּмʼ��������5�*�)�$�%�)�5�B�N�[�g�m�g�b�[�R�N�B�5�5�û����x�l�f�c�f�l�x�����������ûȻ̻ɻûлû����ƻֻ���'�@�M�V�W�R�I�:�'����¿¦�|�|¦¿��������������������¿�T�B�6�.�,�0�b�{ŔŠŭŸž����ŹŊ�n�b�T���������������������������������������ߺɺ��������ɺֺ����ֺɺɺɺɺɺɺɺɻ�	����!�-�:�F�S�l�s�}�x�l�F�:�-�!��0�(�!�����0�=�I�V�b�m�v�r�b�V�I�=�0�л˻ǻлֻܻ���ܻлллллллллн�������!�.�:�;�;�:�4�.�!�������żŷŹ��������������������������ŭťŠŔŇ�{�w�s�z�{ŇœŔŠţūŭųŭŭ�ѿοοѿܿݿ����ݿٿѿѿѿѿѿѿѿѿ������������ÿĿſѿѿѿпȿĿ���������Ě��w�sāĕĘĦĿ����������� ������Ě���
���
��#�+�0�8�0�-�#�������y�u�p�x�y�����������y�y�y�y�y�y�y�y�y�y�<�9�/�'�'�(�/�<�D�H�U�_�]�U�H�B�<�<�<�<�	���	���"�'�,�(�"��	�	�	�	�	�	�	�	�Ľ������½ҽ�������������ݽнļ@�4�4�'�%��'�4�M�Y�f�p�r�t�s�r�f�Y�M�@�ܻл����������ûܻ���� �"�!����������������������������������������������ED�D�EED�EEEEE*E7ECEEEEE7E*EEE 3 ? .  | _ 7 >  L , @  P = = C Y ? t � F 8 B ' v 6 : 8 2 ) P 6 & 4 5 5 v   ~ F A f : 6 " V A F , X N / T ^ q 1 . U . G B 6 - 8 ;  �  .  \  �    �  �    =  �  &  L    �  �  L  �  �  `  o  �    �  �  �  r  j  �  �  /  _  �  �  �  (  `  Q  �  ^  �  9  ,  �  �  .  
  :  d    2    �  F    E  �    �  k  �  �  ^  x  �  +  �;�o<e`B;D���o�#�
�ě��y�#��9X�49X��C���j��1��㽾vɽD����1���ͼ�j�0 ż��\)�<j�0 Že`B�P�`��`B���t���w�\)�t��,1�P�`�u��O߽\)���49X�T������<j�P�`�����w��-����<j�L�ͽq����1�]/�P�`��7L�}�@��L�ͽ�^5�ixսe`B���-�}󶽮{��-�� Ž�-��`BB��A�-�B��B�FB�7BÓB}QB�KB�nB?BB z&B;KB�`B�uB��BRcA�7�B�#A�¾B�B	��B(�B��B&�A�MB-&B�]B)$�B
B�<B"��B%��A�B rB4�B��B��Bi�B ȈB,� BW#B�<B��B	��B
)�B	*B#��B%�B	��B!�jB.��B0�B��B�SB*�B�,BF�B�B
��B��B��B"�~B$��B|#B�iB�A�p�B�WB� BY{B:�BBvB��B�lB<wB�OB pjB@kB*B�B�6B��A�JKB<kB l�B?�B	��B0�B�nB&ÐA��xBABY�B)?�B��B��B"/�B&F�A���B?�B5 +BB?B?�B��B!��B,�iBB�B�GB��B
71B
��B	 B#�1B%JgB	��B!�9B.��B�uB��B��B*��B��B@B��B
��B��B��B"�aB$��B�B��@#A�`mA=�%@?e�A���ACy�A� wA���A���A�^�A��1AM�Ac�%A�9	A���A�R�AG;'A��,@���A�%{AU&�AY�ZAX�A�pA6�OA�� Ao�BA�}�A18�A�rpA�/�A)��A!��BeXB�aAR�>�u�A���A'�[?��@��MA�lf@���@��OA�@BA�[_A�P�@:@���B
��@�dAtyA���A���A|�Aw��A��A��#AmYKA��A]�0A,��@��U@� hA���C�x�?��A�a�A<i'@@��A�~AC�A��A�}A�z�A�A��uAMѹAb�@A���A�yOA�"6AI�A�pE@��6A�?AT��AZ�aAY�A�*A7�A�� Ap�A���A2�A�}�A�m�A*�A!2BI�B��AQ>�ښA��pA'm?���@��A�5Z@��K@�TJA�a(A�P�A��@4$/@xGQBE5@���A��A�N�A�A{Ax��A��A�F�Am!AÐ@A]��A0�X@�t�@���A���C���         
   	   
      9      &   
   
         M                                           
                     &      b         C            u   .               )                     *                           #                                       #   '   #                     !   !                                          %         3            '   -   -                                 /                                                                  #                                                                        !         '               !   -                                 /                           N��SN��O�nN���O�NlhSO;^|Nq��O��N�I�O��Op�O���O���O��yNRg�OBĔN��sOt�N���N�h[O{O�PDO)FOI�N/�]NU��OM�Ng�/N>�O$��O;��Ol�O$\OZZNW9�O�yO�$O�d�O�#nO�4N�W�N�2"O 3O��P-��N$	�NP��O{��O�P�N9��N�Y,O�	KN��|Ns�NV��P8�N��N"I�N�hiN���O�~�O�ZO��O�oO(��  �  M  �  �  �  �  }    
-  �  &  $  b  
�  �  i  a  �  �  �  �  �  �  �  t  �  �  8  *  f  	    �  '  F  o  [  B  �  	  2  q  `  !  s  %  �  k  �  v  0  �  `  &  +  v  =  �  U  8  �  �  �  6  �  !<�t�<�o<D��;ě���o�o�u�49X��`B�o�e`B�e`B��o�,1���
��t����㼣�
��1��j�ě�����/�+������������/��h��h��h������P���o�D���C��C��0 Ž�w�,1�H�9��G��]/�0 Ž0 Ž0 Ž0 Ž49X�49X�49X�<j�@��8Q�D���L�ͽL�ͽT���y�#�aG��}�}󶽃o�y�#���
FO[\hilhe`[OLGEDFFFF""/46//" """"""""""#/6<@HRUVUULH</#)*32)'ou��������������trqo��������������������� "#����������������������������������������������)56??6)DHJUUacmnsuynaULHADD��������������������6BOhmt�����t[B<32+,6������������������������������������������������������������$).6BEOQY\dda[OB6)"$���		����������������������������zz���������zyvvutsrz�������������fgt��������tlg`^ffff������������������������   ����������;<IU`abdbb^UIB<:87;;-/;@@?<;:0//--------��������������������������������������������{uty{���������������������������������������������������������	#0<PUZZWVI<0#
	STYamtz|��znma`UTQQS����������������������������������������<HUanz������znaUHB:<��������������������[ht����������th^[PS[����������������������������������������>BHNZ[a^_[YNMCB=>>>>����������������������������������N[gt����������gNJDGNN[gt����������thNHINX[]gsmig[XXXXXXXXXXX���� ������������
#05>A<:40#
	N[got��������tgZPIIN���������������������#%$ ������������������������������������������������������������������������������������|����������������zx|������������������������������������������������������������EHUYaklaUHB=EEEEEEEE�����"!����������������������������
#*269860#	 ����)5;CIKB,	 �����
#(.&##
������e�a�e�e�p�r�~���������������~�r�e�e�e�e�#�"��#�0�<�=�=�<�0�#�#�#�#�#�#�#�#�#�#�M�C�5�4�4�4�5�A�I�M�Z�Z�f�q�h�j�j�f�Z�M�ֺѺɺĺɺɺֺ�������ݺֺֺֺֺֺ��"������������	�����/�;�<�;�/�"�f�_�b�f�g�s�����������s�k�f�f�f�f�f�f����ĿĹĵĿ�������������������������������������������
��������������������	�����������
��#�.�/�5�6�/�#�#��	�	�	�#���#�#�/�<�C�@�<�:�/�#�#�#�#�#�#�#�#�Z�P�N�A�A�?�A�N�P�Z�g�s�x�y�v�s�q�g�Z�Z���������������������¾ʾξԾԾʾ���������	����	���"�;�G�`�|�z�q�m�T�G�;�.��N�A�5�-�0�9�A�N�g�s���������������s�g�N�s�g�N�I�A�<�G�P�Z�g�s�����������������s���������������������������������������׾������}�g�f�Y�f�s���������������������������������	���"�+�"��	�������������������������������ʼּ��ݼڼּμʼ���ĦĦĲĳĿ������������������������ĿĳĦ�����׾ʾƾ������žʾ������	���	������������	�������	���������׾ɾʾ׾����	���"�'�)��	����������� ����*�5�6�C�F�I�C�6�3�*���������(�4�6�A�J�H�A�9�4�(����������������������������������������������z�y�y�y�������������������������������g�f�g�p�{�}�������������������������s�g����������������������ìêëìù��������ùìììììììììì���������������������������	������ݽнĽ������������Ľн۽����������ݽ��������~��������������ͽͽŽ�����������������������������������������������������������������������������پ��������ʾ׾����׾ʾ����������������¹����������ùϹܹ�����
����Ϲ����������������������������&�"��������������������Ľнݽ���������ݽн�����������&�3�@�Y�~���~�u�L�@������������������ʼּ���ݼּмʼ��������5�.�)�&�)�)�5�B�N�Z�[�\�[�N�M�B�5�5�5�5�������x�l�l�h�l�l�x����������������������������������'�)�)�'������¦²¿����������������¿¦�T�B�6�.�,�0�b�{ŔŠŭŸž����ŹŊ�n�b�T���������������������������������������ߺɺ��������ɺֺ����ֺɺɺɺɺɺɺɺɻ�	����!�-�:�F�S�l�s�}�x�l�F�:�-�!��0�(�!�����0�=�I�V�b�m�v�r�b�V�I�=�0�л˻ǻлֻܻ���ܻлллллллллн�������!�.�:�;�;�:�4�.�!�������ŽŹ�����������������
�����������{�y�t�{�|ŇŐŔŠšŦŠŔŇ�{�{�{�{�{�{�ѿοοѿܿݿ����ݿٿѿѿѿѿѿѿѿѿ������������ÿĿſѿѿѿпȿĿ���������Ě��w�sāĕĘĦĿ����������� ������Ě���
���
��#�+�0�8�0�-�#�������y�u�p�x�y�����������y�y�y�y�y�y�y�y�y�y�/�+�,�/�<�H�U�X�X�U�H�<�/�/�/�/�/�/�/�/�	���	���"�'�,�(�"��	�	�	�	�	�	�	�	�нŽ������ƽнս������������ݽм@�6�4�(�)�4�@�M�Y�f�n�r�r�r�q�f�Y�M�@�@��ܻл��������ûлܼ���������������������������������������������������ED�E EEEEEEE)E*E7ECEDEDE6E*EEE 3 ? .  | _ * J  L , @  K 4 = C Y A t �   ? 5 # v 6 : 8 2 ) P 6 ' / 5 + v   t F 0 4  ) " V A F , X N , 4 ^ q 1 . U + G A * 0 8 5  �  .  \  �    �  �  o  =  �  &  L    W  |  L  �  �  M  o  q      0    r  j  �  �  /  _  �  �  `  �  `  �  �  ^  �  9  �  �  5    
  :  d    2    �  *  �  E  �    �  k  �  �    4  =  +  h  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  F`  �  �  �  s  [  >    �  �  �  �  ^  7    �  �  �    9    M  F  >  7  /  (         
     �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  z  e  L  4      �  �  �  �  �  �  �  u  f  W  D  2      �  �  �  �  x  \  A  .  (  "  �  �  �  �  {  p  f  `  \  N  8  !    �  �  �  �  Y     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  
�  :  a  |  |  u  c  E    
�  
c  	�  	N  �    x  �    �  �  �  �                          �  �  S    �  �  
-  
+  
!  
  	�  	�  	�  	S  	  �  b    �  '  �  �  '    �  �  �  �  �  u  f  [  Q  E  8  *          �  �  �  �  u  F  &  #  !      
  �  �  �  �  �  q  L  $  �  �  �  y  �  �  $            �  �  �  �  �  �  �  �  �  �  �  z  X  6  b  [  S  P  J  :    �  �  �  n  E  2       �  �  �  �  5  	�  
  
Q  
�  
�  
�  
�  
�  
�  
�  
o  
(  	�  	q  �    0  H  ^  �  �  �  �  �  �  �  �  p  ^  L  ?  7  .    �  �  v  ;  �  c  i  c  ]  W  Q  K  E  >  7  0  )  !        �  �  �  �  �  a  \  W  Q  C  4  $      �  �  �  �  �  �  �  �  j  A    �  �  �  �  ~  x  r  k  b  Z  R  J  B  @  I  Q  Z  c  l  u  �  �  �  �  �  }  n  [  >    �  �  U     �  R  �       N  �  �  �  �  �  �  f  K  0       �  �  �  �  l  @  �  t    �  �  �  �  �  x  Y  z  �  �  �  �  �  �  e  =    �  �  �  �  %  �  �  �  �  �  �  �  �  �  n  B  
  �  w  )  �  �  z  �  �  �  �  �  �  �  �  �  �  g  :    �  �  �  �  �  `  �  g  �  �  �  �  �  �  �  �  �  \    �  n  
  �    z  �  .  �  /  N  f  s  q  k  `  S  E  8    �  �  ~  7  �  Z  �  !  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    y  r  �  �  �  �  �  �  �  �  �  �  �  �  |  o  b  T  O  L  I  E  8  2  +       �  �  �  �  �  �  �  �  �  y  f  ^  V  R  O  *  &  #  &  (  (  '  &  &  $           	    �  �  �  z  f  Y  M  @  1  !    �  �  �  �  u  N  %  �  �  �  g  5    	    �  �  �  �  �  �  �  �  �  �  o  ^  P  E  ;  7  7  7        �  �  �  �  �  �  q  a  Q  @  +    �  �  �  �  �  �  �  �  �  �  �  �  {  f  N  5    �  �  �  �  Q  �  \   �  �      $  #      �  �  �  h    �  w    �  ?  �  \    �  !  @  F  F  B  8  '    �  �  �  .  �  s  
  �  �    (  o  h  b  \  U  O  H  C  =  8  3  -  (  "         �   �   �  �    .  [  R  >  #    �  �  D  �  H  
�  	�  	  �  �    w  B  4  %        �  �  �  �  �  �  �  f  =    �  �  �  a  �  �  �  s  Y  ;    �  �  �  �  �  |  d  C    �  �  Z  3  �  �  �  �  �  �  �  ~  .  �  Q  �  c    �    �  �  ;  �  2  -  (  !      �  �  �  �  �  w  q  k  `  N  =  #     �  J  \  k  o  m  e  [  P  ?  -    �  �  �  j  '  �  �  3   �  �  �  �  �  8  A  T  `  X  >    �  �  �  A  �  �  g  �  8  �  ~  	  	�  
   
�    q  �  �  
      �  v  
�  	�  �  ,    �    E  ^  k  s  n  X  .  �  �  *  �  �  �  `  /  �  t  �  %      �  �  �    G  
  �  �  �  �  �  h  F    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  d  H  +    �  �  k  d  ^  X  Q  K  E  @  ;  7  3  0  +  '  %  &  '  
  �  �  �  �  �  �  �  �  x  a  I  -    �  �  �  r  Y  7    �  �  v  k  ]  N  F  O  c  a  S  B  $  �  �  w  '  �  4  �  �  �  0  +  &  $  #  &  )  +  '    	  �  �  �  �  R    �  �  U  �  |  n  ^  H  3      �  �  �  �  �  k  N  ,  	   �   �   �  `  _  S  G  4    �  �  �  �  _  *  �  �  `    �  [  �  �  �  �        �  �  �  �  {  @  �  �  b  
  �  I  �  j  �  +  )  &  $  !              
    �  �  �  �  �  �  �  v  n  g  `  X  Q  J  B  ;  4  ,  #         �   �   �   �   �  =  4    �  �  �  �  �  �  �  l  n  d  ?    �  r     �    �  �  �  �  �  �  z  o  e  W  H  9  &    �  �  �  �  Z    U  I  <  0  $    	  �  �  �  �  �  �  �  m  R  4     �   �  �    #  1  4  8  5  +    �  �  �  M  	  �  J  �  R  �    �  �  �  �  �  �  �  �  �  x  ^  D  (    �  �  �  �  y  \  }  �  �  }  l  U  7  
  �  �  i  .  �  �  �  �  y  Y  7  )  �  �  �  �  �  �  �  �  �  �  ^  -  �  �  ?  �  -  �  �  I    )  4  6  1       �  �  �  f  +  �  �  A  �  m    �  �  �  �  �  �  �  t  S  0    �  �  �  H  	  �  }  6  �    r        �  �  �  �  �  `  '  �  �  �  4  �  �  &  @  P  V