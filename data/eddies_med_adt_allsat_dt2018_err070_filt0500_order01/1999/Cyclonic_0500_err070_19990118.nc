CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��G�z�        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�Bu        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��
=   max       <�j        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F~�Q�     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ҏ\(��    max       @vc�z�H     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P`           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @���            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �'�   max       ��o        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�b�   max       B1�        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~R   max       B0�{        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >`��   max       C���        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >S��   max       C�        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          j        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P��        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����o   max       ?����        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��
=   max       <u        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F}p��
>     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �θQ�     max       @vc�z�H     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P`           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @���            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�_��Ft   max       ?����        W�   )         %   i            E                        F   +            1                           -   (         7      *   "                           	   
            (                                 B   O��N��jOG�QPRY�P0��N��O�lNw��P�BuN��N!��O(_N�N�&O��O|�P11xP
}O�c�N���M���O��lO��EO��OtI7O�O1�N�LN��+O,O��.O��OrJ�O7e�P�NI6�O�EO�&�OOOYN��YO��N�N3\�Nd��N&�OAN���NS�Ou]�N�@O�ޱO��hO�`�ON�N��CN9�O���O'��O6fO ?�N1��N�h/PF�Ny�<�j<�o<t��o��o�ě���`B��`B��`B�o�t��49X���
��9X��9X�ě��ě��ě���������������/��h��h�����������o�o��w�,1�49X�49X�8Q�<j�D���D���L�ͽT���Y��Y��]/�aG��e`B�e`B�q���q���}󶽃o�����������\)��\)�����������-���-���w���
������
=������ ���������������������������wz�������������{zpqw)+6BOh���������O6)))B[r�|vjgjb`ZPB4$��������������������NRY[gt�����zogd[PNNN����������������������#<n�����{bI<0
����������������������� �����������"#/<HU\cea_ULH<5/# "�����������IN[gmsmg][YNKFIIIIIIy����������������xyylnw�����������znhdel�������
��������#*5C\hu����uh\C6*#!#������#/-#
�������������������������������������������������������������������)6B[jkf^[OB6)�����$%����������������������������������	�������~��������������~{{|~8<BHUUZ\\UOH<8888888~�����������������}~T[gtw���������tg[[RT������������������������������������w{������������spsxwz�������������zvtuvzi�����������������ji{���������{{{{{{{{{{HUaz�������znaUHB@BH[f����������}ytg[UU[������ 

�������������������������������
#/</#
�����lmnz��������zmllllll��������������������
!%#



��������������������(0<ILOQTID<60'((((((��	
	������������������������������������������������������� !��������LR[gt~�������tg[NJJL����������������������������������������#)049830'# 
#&(+)#$5BN[acZNB5)�)6BDEB75*)���	 �����)+59BFHB95)nt������tonnnnnnnnnn��������������������'5Bgt��tjgdgmf[N5#&//25/+#!��
����������������#�/�<�B�I�E�<�/�#���������������������������������������Ѿ(�$������(�4�A�J�M�W�[�Y�M�K�A�4�(�T�;�*�����
��.�T���������������m�T�S�A�:�.�:�F�l�������ûл��ܻӻл��l�S�I�I�I�U�U�U�b�n�o�n�b�U�I�I�I�I�I�I�I�I�����������~�}�������������¿ĿͿĿ������H�A�<�:�:�<�?�H�U�V�Y�X�U�O�H�H�H�H�H�H���������b�R�@�I�f��������������������;�:�/�%�/�;�H�T�a�e�h�a�T�H�;�;�;�;�;�;��u�y������������������������)�(�!� ����)�6�?�B�E�O�R�V�O�J�C�6�)�H�=�B�H�T�a�d�a�Y�T�H�H�H�H�H�H�H�H�H�H�M�K�F�K�M�Z�f�j�s�s�s�j�f�Z�M�M�M�M�M�M�A�8�4�(�"�����(�4�A�K�M�N�P�U�M�A�A�	�������������	���"�/�2�6�;�/�"��	��ܻлȻлһ�����4�M�Z�`�\�S�4����ʾ����������������׾�����������ʺV�M�K�G�H�e�r�����������������~�x��r�V���ݿѿĿ����������ĿͿѿݿ������m�l�m�y���������������y�m�m�m�m�m�m�m�m���ѿ������������ѿݿ����������������������������������������������������������ż�����!�,�9�6�.�!�����ּʼ�àÓ�z�t�n�a�a�n�zÇÓìù��������ùìà���������������������	���"�#���	�����Z�V�G�D�L�N�Z�g�s�����������������s�g�Z�(� �����(�5�A�E�A�<�9�5�(�(�(�(�(�(�M�D�A�=�A�C�M�R�Z�f�s���x�v�s�f�a�Z�M���������������	����#�"����	�������׾ľ������׾�����&�$���
�	���������� ���*�C�J�\�_�q�h�\�O�C�*��������"�@�M�f�r�v�r�l�b�Y�@�4�'�������������������������$���������č�w�g�5�+�?�OāčĦĳĿ��������ĿĪĚč�s�l�p�s�y�����������s�s�s�s�s�s�s�s�s�s���������ùϹ�������������Ϲù�������������������������)�5�B�K�K�6�������������������������������������������ż�߼ڼ޼��������������������пĿ¿ĿȿϿѿڿۿݿ�����������ݿ��������������������������������������������������������������������������������ؾf�c�Z�N�M�Z�f�s�x�t�s�j�f�f�f�f�f�f�f�f�����������'���������������������������������������(�����������߽��������������ĽɽĽ��������������������������������	���	��������������������ƚ�u�h�\�R�O�F�O�Q�\�hƁƎƓƞƩƮƧƣƚ�e�b�Y�U�Y�e�r�|�t�r�e�e�e�e�e�e�e�e�e�e������!�:�S�p���������}�S�G�:�.�!���
� ����
��#�0�<�I�U�W�b�a�\�C�0�#��0�*�$�&�4�I�M�b�n�{ŇşŔŒ�{�b�S�I�<�0�ֺͺ����������������ɺֺ��������ֻS�H�F�;�F�M�S�_�l�x�}��������x�l�_�S�S�����������������ûλɻû����������������������j���������������������������������5�*�(����$�(�5�<�N�O�Z�]�a�Z�N�E�A�5������������$�)�6�B�M�B�:�6�,�)����O�J�O�Y�[�e�h�o�tāċčĎĊā�~�t�h�[�O²¬§¯²¶¿����¿²²²²²²²²²²��������������������������������������Ó�a�L�D�7�0�8�H�U�a�nÇàìùþýöìÓEiEcEiEuEvE�E�E�E�E�E�E�E�EuEiEiEiEiEiEi   +  . E ] V W M j : E N , P ] & 6 } X P + A s K B * 4 H X D < [ + V t R q 8 r p e 0 4 h : K @ N K k  w Y   � I $ u ] ? K X i  E  �  �  �  A  P    �  �  �  5  �  5  �  P  u    Q  �  �  4  �  �  6    E  |  �  $  �    �  S  �  �  �  �  9  �  �  �  �  O  y  c  :  �  a  -  &  �  l  �  �    N  >  l    :  ^  �  �  ���t���o��C�����;d�o���ͼ��
�����D����o�o��9X��`B�\)�o��vɽ�7L�L�ͼ���`B����0 ŽP�`�H�9�\)�<j�t��']/���㽡���aG���������D����-���+�y�#��C��q���q���m�h�q����hs��o��O߽��w�����1��
=���罡����E����P����ȴ9��^5������{��Q�'��B(B� BE�B��BB,�B	@hB ~ B&��B�B\_B�B��BÈBUBc�BGTB1�B"��B*yB*ĤB�~B�B-�^B�B�Bm�BUAB�kB	�$BE�B�uB)��B t"B�<B�By�B
;B�[BOBjeB BF�BjB$x�B+�B&D	A�b�B�B9�B�B	;mB�B�6B%.B%8kB�0B�0B;�B��B
*�B
�BQ)B�]B?B��B�IB�oB�UB>zB	>@B �B' HB�[B@`BHZB��B�~B��B7�B��B0�{B$B*D�B*��B�B¼B.<�B�B�B=�BF�B�B
/�BS7B?�B)�SB @�B�B�B�B	�$B�kB��B�jA��TBK:BE3B$�XB9�B&�~A�~RB?�B>�B�B	@�B�B}�B$�uB%<dB��B¥B?�B4SB
:IB
�xB�~B��A���A�R�A9=�Ak��@���A��As�]A���A���A���AFJ�A�P_A�*)A?�A9)�A�q�@ċAS~-@
gAz�GAn(�A}S�A���A�A��A�v�A�փA��A?�A[@�AV*6A���@�AzA��Aݿ�A�O>`��A�#�A�M�A@�A|��A�pA�
A@}�A2Y�A��$A#9�A�wBn?�@�A�@A�95A�a�@8��@���@��A��A��A�I�A���A�̗BL$A�DC���A��rA�vMA9�Ak�@�رA���As5�A��A���A���AF��A׀A��.A?f7A8��A��D@ˡ0ATǵ@�dAz�3An$�A}�A�t�A	W�Aʓ�A��[A� �A��A?5wAZ��AUhA�~S@�Aғ2Aܣ[A���>S��A�{�A�}fA��A}9A�	�A��A@�A4&�A��SA# �A�~�B�P?�}�A�9A�2�A��@>*�@��@��A��mA���A�|�A܊�A��BBaJA��C�   *         %   j            E                     	   F   ,            2                           -   )         8      *   #                           	               )                                  B               -   1            ?                        +   '   +               %                     #            -      #   %                                       !                  '                  +                              ;                        )   '   +               %                                 )                                                !                  '                  +   Oe��N�f�O��N���O-�kN��O �	NS�P��N��N!��O(_N�N�&O��N�0%P�GO�2VO�c�N���M���Om�cO��EO��O`c!O�N�;MN�LN��+O �O�W�OE�%OR+SO-��P ��NI6�O�Oy�OOOOYN��YN�[,N_+�N3\�Nd��N&�OAN���NS�OQ�N�@O�ޱO>�KO�`�OCN�3@N9�O���O'��O6fO ?�N1��N�h/PF�Ny�  1  �  q  �  	�    >  �  @  �  �  �  ]  �    J  �  W  N  �  W  �  �  �  �  �  �  �  �  �  Q  �  �  �  	Q  0  �  �  �  �  �  +    �  k  m  &  =  t  M  �  �  �    �  O  �  �  f  	�    �  
�  G<u<D��;��
�����y�#�ě��t��o�D���o�t��49X���
��9X��9X���������ͼ�������������w��h��h�����C������+�'L�ͽ0 Ž8Q�D���8Q�T���aG��D���L�ͽ]/�]/�Y��]/�aG��e`B�e`B�q���y�#�}󶽃o���㽅���+��t���\)�����������-���-���w���
������
=����� ��������������������������������vzz��������������|zvghkpt��������tohgggg369BO[_fhge[QOHB>:53��������������������Z[gnt{~wtmg`[UOPSZZ���������������������#0Ib{�����{]I0������������������������ �����������"#/<HU\cea_ULH<5/# "�����������IN[gmsmg][YNKFIIIIIIy����������������xyygnpz���������zznmhgg������ ���������$+6COhu����uh\C6*$!$������#/-#
�������������������������������������������������������������������)6B[jkf^[OB6)�����$%����������������� �����������������	����������������������~8<BHUUZ\\UOH<8888888~�����������������}~U[gmtv��������tg\[SU�������
	�����������������������������{�������������|uru{{wz{�����������~zwuuwo�����������������oo{���������{{{{{{{{{{HPUanv�����znaUHDBDHX`gt���������vtgbXWX������ 

�������������������������������
"
���������mmpz������zmmmmmmmmm��������������������
!%#



��������������������(0<ILOQTID<60'((((((��	
	������������������������������������������������������� !��������MQX[gqt|���ztg[WQMM����������������������������������������#'02760/##&(+)#$5BN[acZNB5)�)6BDEB75*)���	 �����)+59BFHB95)nt������tonnnnnnnnnn��������������������'5Bgt��tjgdgmf[N5#&//25/+#!�������������
��#�/�<�D�A�<�5�/�#��
�������������������������������������������4�(�(�����$�(�4�A�C�M�S�W�R�M�A�=�4�y�y�m�`�[�U�]�`�m�y�����������z�y�y�y�y�l�e�_�Y�R�P�S�_�l�x�����������������x�l�I�I�I�U�U�U�b�n�o�n�b�U�I�I�I�I�I�I�I�I�������������������������ĿɿĿ����������H�F�<�<�;�<�A�H�U�U�X�W�U�K�H�H�H�H�H�H�������f�]�Q�I�K�U�g��������������������;�:�/�%�/�;�H�T�a�e�h�a�T�H�;�;�;�;�;�;��u�y������������������������)�(�!� ����)�6�?�B�E�O�R�V�O�J�C�6�)�H�=�B�H�T�a�d�a�Y�T�H�H�H�H�H�H�H�H�H�H�M�K�F�K�M�Z�f�j�s�s�s�j�f�Z�M�M�M�M�M�M�A�8�4�(�"�����(�4�A�K�M�N�P�U�M�A�A���	����������	���!�"�/�/�0�/�"�����ݻӻٻۻ߻����4�M�W�]�Z�P�4�'�����ʾ����������������׾���
����
����ʺV�M�K�G�H�e�r�����������������~�x��r�V���ݿѿĿ����������ĿͿѿݿ������m�l�m�y���������������y�m�m�m�m�m�m�m�m�ѿĿ����������Ŀѿݿ��� �������ݿ������������������������������������������������ż�����!�,�9�6�.�!�����ּʼ�ÓÇ�z�u�n�a�n�zÇÓê����������ùìàÓ���������������������	���"�#���	�����g�[�Z�N�L�K�N�Z�f�g�g�s�����������s�g�g�(� �����(�5�A�E�A�<�9�5�(�(�(�(�(�(�M�D�A�=�A�C�M�R�Z�f�s���x�v�s�f�a�Z�M����������������	���"�!����	�����׾ʾ��������̾׾�����������������������*�6�A�T�Z�O�M�C�6�*������&�@�M�Y�f�p�n�j�`�Y�M�@�4�'�������������������������#��������č�{�i�7�.�B�O�[�hāčĦĳĿ������ĿĚč�s�l�p�s�y�����������s�s�s�s�s�s�s�s�s�s���������������ùϹܹ�����ܹϹù�������������������������)�5�<�>�5������������������������������������������ż�߼ڼ޼��������������������ĿÿĿƿʿѿѿݿ������ݿѿĿĿĿ��������������������������������������������������������������������������������ؾf�c�Z�N�M�Z�f�s�x�t�s�j�f�f�f�f�f�f�f�f�����������'���������������������������������������(�����������߽��������������ĽɽĽ��������������������������������	���	��������������������ƗƎƁ�u�h�\�T�P�W�\�hƁƎƑƛƦƪƧƝƗ�e�b�Y�U�Y�e�r�|�t�r�e�e�e�e�e�e�e�e�e�e������!�:�S�p���������}�S�G�:�.�!��#��
���
���!�#�0�<�I�N�U�O�I�<�0�#�0�*�$�&�4�I�M�b�n�{ŇşŔŒ�{�b�S�I�<�0�ֺϺɺú����������ɺֺߺ��������ֻS�J�F�?�F�P�S�_�l�x�x���}�x�l�_�S�S�S�S�����������������ûλɻû����������������������j���������������������������������5�*�(����$�(�5�<�N�O�Z�]�a�Z�N�E�A�5������������$�)�6�B�M�B�:�6�,�)����O�J�O�Y�[�e�h�o�tāċčĎĊā�~�t�h�[�O²¬§¯²¶¿����¿²²²²²²²²²²��������������������������������������Ó�a�L�D�7�0�8�H�U�a�nÇàìùþýöìÓEiEcEiEuEvE�E�E�E�E�E�E�E�EuEiEiEiEiEiEi  (  # & ] M ^ J j : E N , P L # 7 } X P ( A s E B * 4 H V B ; K + T t - J 8 r D R 0 4 h : K @ K K k " w O  � I $ u ] ? K X i  �  �  M  �  {  P  ,  �  �  �  5  �  5  �  P  �  �  ?  �  �  4  �  �  6  �  E  �  �  $  }  i  �  �  p  �  �    0  �  �     �  O  y  c  :  �  a  �  &  �  �  �  B  �  N  >  l    :  ^  �  �  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �    -  /  #    �  �  �  f    �  ^  �  D  �  �    �  �  �  �  �  �  �  �  �  �  �  c  6    �  �  [    �  O  �  :  [  l  q  n  d  V  B  *  	  �  �  �  L  
  �  \  �  U  �  S  Z  O  F  5  '  ?  F  V  q  {  �  �  �  �  �  V    �  %  �  z  /  �  �  	  	'  	Q  	�  	�  	�  	�  	�  	P  �    �  �  g  1                                 %  *  .  3  7  *  =  >  9  +      �  �  �  �  �  l  L  )  �  �  :  x     �  �  �  �  �  �  �  |  Z  4    �  �  m  3  �  �  �  [  !  ;  ?  8    �  ~  $  �      �  T  	  �  �  P    �  ?  @  �  �  �  �  �  y  n  b  V  J  ;  *      �  �  �  �  �  }  �  �  �  �  �  �  �  �  �  z  j  X  =  "  �  �  �  g  0   �  �  �  �  �  �  �  ~  l  P  1    �  �  p  +  �  �  0  �    ]  U  M  F  >  7  /  (            �   �   �   �   �   �   �   �  �  �  �  �  �  �  {  k  Y  H  4      �  �  �  �  w  R  .          �  �  �  �  �  �  v  c  P  8    �  �  �  �  q  C  >  9  <  D  J  I  H  B  ;  3  )      �  �  �  �  M   �  �  �  �  �  o  H  '    �  |  O  �  �  �  s  !  �  �  �  �  T  S  L  C  4  !    �  �  �  J    �  �  e  )  �  k    Z  N  /    �  �    )  .  0  -  6  *  	  �  �  �  F    �  3  �  �  �  �  �  �  z  p  f  \  P  @  0  !     �   �   �   �   �  W  N  D  ;  1  '           �   �   �   �   �   �   �   �   �   �  x  �  �  �  �  �  �  �  �  �  <  �  �  #  �  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  e  C    �  �  L   �   �  �  �  �  �  �  �  �  �  �  |  i  S  6    �  �  Q  ;  �  Q  �  �  �  �  �  �  p  U  7    �  �  n  %  �  �  v  4  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  c  S  B  :  [  v  �  �  �  �  ~  h  P  4    �  �  �  {  E  
  �  �  �  �  �  �  �  �  z  j  Y  E  2      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  b  F  '  �  �  �  L   �   �  �  �  �  �  �  �  �  z  _  ?    �  �  O  �  �    �  <  �  �    4  I  Q  N  I  <  )    �  �  �  _    �  E  �  $  �  	  M  {  �  �  �  �  �  �  j  2  �  �  J  �  {  �  L  x  �  �  �  �  �  �  �  �  �  y  `  E  ,      �  �  �  �  �  \  �  �  �  �  �  h  D  !  �  �  �  ]  �  �  (  �  0  �    1  	'  	E  	K  	&  �  �  �  �  �  X    �  R  �  [  �  �  �  �  �  0  -  +  )  &  $  !          
    �  �  �  �  �  �  �  h  �  �  �  �  �  �  �  �  i  >  
  �  |  %  �  A  �    F    	  .  �  �  �  n  O  )  �  �  �  ;  �  �    �  �  �   w  �  �  t  [  @  "    �  �  �  W  !  �  �    K    �  f   �  �  w  `  :    �  �  �  O    �  �  ~  K    �  �    �  t  0  D  �  �  i  @    �  �  �  m  I  )    �  �  �  ^  �  �    !  &  +  "    
  �  �  �  �  �  y  `  F  -    �  �  �    �  �  �  �  �  �  �  �  �  �  z  k  ^  T  J  >  (    �  �  �  �  �  �  |  m  ^  O  @  2  &         �  �  �  �  �  k  a  X  O  F  <  2  (      
     �  �  �  �  �  �  �  k  m  f  ]  O  9    �  �  �  �  n  >    �  �  d    �  �   �  &    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  =  !    �  �  �  p  K  &    �  �  k  ,  �  �  z  =  �  �  i  r  t  g  Q  8      �  �  �  �  ]  ,  �  �  G  �  _    M  D  ;  2  )         �  �  �  �  �  �  �  �  �  �  x  j  �  �  �  �  �  r  T  6      �  �    Y  5    �  �  �  m  W  �  �  �  �  �  �  �  �  �  �  X    �  �  F  �  u  �  x  �  �  �  �  z  `  B       �    �  �  �  l  ?    �  �  �  �     �  �  �  �  �  �  �  �  w  \  ?  !    �  �  �  }  F  �  �  �  �  �  �  }  r  _  R  C  1  "    �  �  J    �  P  O  A  3  %      �  �  �  �  �  |  Z  8     �   �   �   �   �  �  �  �  �  �  �  y  Y  6    �  �  �  H  �  �  X  )  �  �  �  �  m  J  %  �  �  �  �  V  %  �  �  �  D  �  f  �  �  !  f  +  �  �  �  �  z  B      �  �  �  q  /  �  �  X    �  	�  	�  	�  	�  	X  	)  �  �  �  _    �  p    �  O  �  y  �  �                      
           �  �  �  �  �  �  �  r  [  C  +    �  �  �  �  �  �  l  T  ;  !    �  �  
�  
�  
\  
&  
P  
+  	�  	�  	{  	  �  H  �  �  !  �  �  �  �    G  "  �  �  �  �  W  )  �  �  �  [  *  �  �  �  �  �  e  �