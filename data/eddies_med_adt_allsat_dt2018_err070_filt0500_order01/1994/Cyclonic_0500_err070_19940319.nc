CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�?|�hs       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��d       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��%   max       <o       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�=p��
   max       @F\(��     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vi\(�     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P�           �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @��           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��Q�   max       ;D��       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�   max       B4ͧ       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�n�   max       B4��       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       > �f   max       C�(e       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >A�c   max       C�'�       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          S       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�;K       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��_o�   max       ?�$tS��N       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��%   max       <o       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�G�z�   max       @F���R     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vip��
>     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P�           �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?   max         ?       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*0U2b   max       ?�X�e,     �  Zh                           
         S   ;   &         	   
      	               0                  	      
                   +      
                                                          	               	      	N��	Nh�oO-�tN`��N�0�OR�N�G�NʸuO��O��)N!�{P��dP!4�PV��O���N![�N���O��P*xmOKO���O�)�NL�O�O��N�[}Otk�OO�NL�O��WO9�O�>�O�rOe�!O
ƨP�iOF�N�P ��N�?�N]m�O��N=��OB9tM��N1�
O��qNhdO�=�P	NOѺO��OD�NX%N5cuO��O��HN��N%GN��NKO�g�N�׌N�+AN��O%�N-��<o;��
�D����o��o���
���
���
�ě��o�o�o�#�
�#�
�#�
�D���D���D���D���T���e`B��o��C���t����㼛�㼛�㼣�
��1��9X�ě����ͼ��ͼ���������/��/��h��h��h��h��h���o�+�C��t�����w�#�
�',1�,1�0 Ž<j�D���D���L�ͽL�ͽL�ͽL�ͽP�`�T���T���y�#�y�#��%��������������������()5AA>5) ((((((((((#/<CGHTRH</#"!!	&)--+)��)���������������� 

������������������������������������������������������
������������������������������#0n�������{<����4 
�����
#/<HLABHG4���������������������*6COXdgiW6*����������������������#/1<HHIH</(#��)BB6)(�����')1=BO[hqttO?)
����������������������������������������6O[h����vohM6)��������������������RUansz{��zynlaUUOLRRVamz��������zmkc`TRV��������������������[`gt����������tg[VT[��������������������ABORYXOJBAAAAAAAAAAA��������	 ������������������������z���������������������)57;875/)$����������������������������������������������������������}�()6;BEOSTOKB6,(%&&&(?BCFN[ggniig^[NMDB??��������������������MOV[chnjh_b[XVOMMMMMcht����xthgacccccccc������������������������������������������������������������	
  
									��������������������<CUanz�������znaH97<�� 

���������ru�������������xvpnr����8;>70)�������������������������ABN[gpqkgc[YNGBBA@AA�� )-)'�����@BLNPWZSNBA=@@@@@@@@#,08<@<0(#�#&06;0#
���;@Ibw{|{{���{b\LI=;������������������������ ��������������FHTUWVVUUHECFFFFFFFFLO[chijh`[OGLLLLLLLLdm������������i_]]bd��
"##
 ���"#/3<>C<94/#"" !""""����������������������������������������7<HSQHG<967777777777�#�������#�-�/�<�?�<�;�9�/�#�#�#�#�=�<�5�=�I�V�Y�Y�V�I�=�=�=�=�=�=�=�=�=�=�H�D�<�6�0�/�<�H�K�U�a�h�l�p�o�n�a�\�U�H��غ׺��������������������!�����!�&�-�:�F�I�M�F�?�:�-�!�!�!�!��w�t�z������������������������������s�k�g�d�d�d�g�m�s���������|�����������s�ʾž������������������������ʾϾԾξʾʾ������������������ʾ׾�����׾ѾӾξ��M�E�A�K�W�f�s�������������������s�Z�M���	�����������	���������������s�\�T�9�8�H�Z�����������������������t�[�B�)��������������B�[�y�`�O�F�G�P�T�`�y���Ŀ޿������࿸���y�;�.�"����������	��.�I�L�J�M�M�G�;����������������������������������)�)�(�)�)�/�6�6�B�C�I�D�B�@�6�*�)�)�)�)�T�O�N�O�N�T�T�U�a�j�m�s�z�}�|�z�t�m�a�T���`�G�.�"��	��#�G�q�y�������Ŀȿ��������ܹ̹ŹŹϹڹܹ���������������B�6�/�$�&�)�6�@�B�O�[�c�h�yĀ�~�t�[�O�B�����}���~�~�q�|�����������������������������(�5�A�G�A�<�5�(���������(�#��#�(�3�4�A�M�Y�Z�\�Z�R�M�M�A�4�(�(�g�Q�I�F�L�Z�s�������������������������g�A�7�5�3�5�:�A�N�X�Z�b�d�c�b�Z�N�A�A�A�A���۾ܾ�����	����"�%�!���	����ŠŔŔňŎŔŘŠŭŹ��������������ŹŭŠ�l�i�f�l�x�������~�x�l�l�l�l�l�l�l�l�l�lìàÓÍÇÃ�|�zÄÇÓàæìù������ùì�:�/�2�6�;�@�T�m�z�~�����z�m�c�T�S�P�H�:ĳĭĮıĿ�������
��#�+�#��
������Ŀĳ�������� ����(�,�5�>�A�J�A�3�(���_�h�d�c�d�l�x�x�|���������������x�l�d�_������r�f�d�c�f�r���������������������ƣƚƘƢƳƺ��������)�*�$���������Ƴƣ�f�`�]�f�k�s�����������ľ�����������s�f�������������)�-�6�:�8�6�)�#���ֺ������������ɺ����-�F�X�\�U�F�!���ֹù����������ùϹչܹ����ֹܹϹùùùú�����'�*�3�4�4�3�'���������2�1�9�<�@�B�L�Y�f�r���������~�r�e�Y�5�2�������������ĿͿѿҿѿĿ��������������������������������*�8�9�/�*���������׾վ׾�����������������.�"�"��!�"�.�4�;�?�;�5�.�.�.�.�.�.�.�.�����������������ĿѿԿ׿ۿ�����ѿĿ��������(�4�7�?�4�-�(�������ŭŠŔőœšŭŹ��������������������Źŭ����ùíÇ�v�o�zÇà�������������������������������������������������������������/�/�%�&�,�/�<�H�N�U�`�a�i�a�U�Q�H�<�/�/�M�L�>�4�3�4�;�@�M�Y�a�f�q�r�w�s�r�f�Y�M�������*�6�=�7�6�*���������!� ��!�%�-�3�:�?�E�:�2�-�#�!�!�!�!�!�!�ܻû����������������ʻлܻ�޻ݻ���ܻ�л»��ûȻлܻ���'�4�9�.�'��������/�)�&�/�<�B�H�L�H�<�/�/�/�/�/�/�/�/�/�/D�D�D�D�D�EEEEE	ED�D�D�D�D�D�D�D�D��������������ùϹй׹Ϲù����������������Ľ½��������ĽɽнսӽнĽĽĽĽĽĽĽ�¦¦²¿������������������¿²¦E*E#E!E*E2ECEPE\EiEuEwEwEiEhE\EPECE7E3E*E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���������$�&�$��������������������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� 0 F ) 7 4 ' k O Z F � I [ 6 b L w r Z d 2 u Q 7 _ 3 A 7 8 G a l 7 ^ 3 R N  ` C ) 9 E 5 i & T D  Q # - F F y _ P Q ^ k \ & n N g % D    �  d  s  l  �  H    �  a  $  �  e  P  �  �  J  �  �  i  =  
  }  �  ,  �      �  '    �  �  V  $  8  �  �  	  �  �  h  �  l  �  2  K  �  �  �  �  W  4  e  o  �  p  �    n  J  G  �  g  �  `  >  D;D��$�  ��`B�o�D����1�#�
�49X��o���D����Q콍O߽D���+��o��1��9X����1�\)��㼣�
��󶽇+���ͽ'����ͽL�ͽ+��P�\)�49X�,1�q���0 Ž49X��t��0 Ž�w�u�o�<j�t��#�
��C��8Q�]/��+�P�`�Y��u�@��H�9�Y��y�#�aG��e`B�q���Y����-���
��O߽�O߽�O߽�t�Bs�B�B1B`tB�#B6B�B!�B4ͧB�B!b]B&�7B|~B+M�B0!�B ��B9#B%nB�BcgB�{B��BfVB|iA�B��B	�BߵB�*B]�BT�Bb;B�CB U�B �B�B��B�B0BK�B*B �B"�BhB.�B��BJ�B$7"B
��B�BB�gB$�B;�B%hVB$�/B'�{B
��B�5B��B1�B
`jB�zB)5BZ�Bh�B	�Bv3B	B'!B{jB��B?2B�B!��B4��B�;B!kgB&�kB��B+>;B0BSB фB>�B6*B>�BD�B�B>�BM�B?�A�n�B�B	��B��B��BĆBĸB�JBE2B �AB =*B�BhB��B@ BB;B8[B ��B�JB?�B@�B�bBA�B$@ B9kB?�B5[By�B�B=�B%;;B$�B'��B
�%B<�B?�BB�B
A�B6�B?wB>�B�zB��A��vBT�A�B�@J��@u&�AH.EA�kHAMD�ANLACMA�y�A�d�A��=As�NA^fvA���A�QA�j�Ak�?�A�`�A�^}A�ɾA:�A�Z�A�S�AZGA�"�@�S�A˯A��YA��XA��u@�_�@�eB��AG��A�c@]H�>lo�?��f?�.Ax&�A��nAV)�A`�*AyaA63MA��@A�O�A�S�A�0Q@�
�A��@u
J@�X+@�E�A�U�C�VI> �fA'.�A��0C��"C�(eB	
A�l�C��6A�z�B�A�[@H&�@twAGqA��AM�BAN��AC�A���A���A�z�Av�`A]*A�[�A��A�Ak��? dA�~IA�|�A�{%A:�sA�0A���AZ�8A��@��BA�z�A���A��A�@�@��BABAI�eAՐ@T J>G�?�+Z?�m�Ax��A��pAT��A`��Ay%oA8X�A�}�A͡�A�y�A��F@�_A�t�@le�@���@��8AÀNC�W�>A�cA'A�:C��TC�'�BˏA�QC���               	                     S   <   '         
         	               0                  
               !         ,                         !                                       
               	   	   
                                    ;   1   5   #            3         -         +                                 +         )         %               !         '                     !                                                                  7      1   !            3         '                                                   #                                 '                     !                              N��	Nh�oN�/�N`��N�0�Nx��NR�N�fsN��	O��)N!�{P�;KO���PF�O�мN![�NX��N��P*xmOKO�iO��DNL�O�O&\�N���OD�O7�NL�OD��O9�O�>�O�rO'��N��OǠ�OF�N��kO�Nl�N]m�O���N=��OB9tM��N1�
O��NhdOY�PP	NO t^N���OD�NX%N5cuO��O��HN��N%GN��NKO�g�N��N��N��O%�N-��  +    �    �  �  �  �  /  �  �  �    U  �    W  �  �     X  �    K  ]  "  b  {  Y  y  M    �    �  *  �  �  �  �  �    N  �    �  =  ;  E  �  �  �  �  2  �  <  f  �  �  �  J  Q  	�  �  �  �  �<o;��
�T����o��o�49X�ě��ě��t��o�o�����/�D���D���D���T���T���D���T����9X��t���C���t���w���
���ͼě���1��/�ě����ͼ��ͼ�h��`B�o��/���\)�o��h�����o�+�C��L�ͽ��'#�
�,1�49X�,1�0 Ž<j�D���D���L�ͽL�ͽL�ͽL�ͽP�`�Y��]/�y�#�y�#��%��������������������()5AA>5) ((((((((((#/;<A<;/#!!!	&)--+)���	
��������������������������������������������������������������������������
������������������������������#0Un������{<0
���� 
#/:=>=3*#
�������������������������
*6COT_deTB6*���������������������#-/<FF</+#��&)0)%�����')1=BO[hqttO?)
����������������������������������������)6O[h����sohB6)��������������������RUansz{��zynlaUUOLRR^aimxz������zma`\[\^��������������������Y[`gt��������tgg[[YY��������������������ABORYXOJBAAAAAAAAAAA�����������������������������������z���������������������)57;875/)$������������������������������������������������������������()6;BEOSTOKB6,(%&&&(@BEKN[dgmhhg\[NNEB@@��������������������OOY[^hlhh[POOOOOOOOOcht����xthgacccccccc������������������������������������������������������������	
  
									��������������������QUacnsz|~zwnaWUIJKQQ�� 

���������pty��������������ytp����8;>70)�������������������������ABEN[fglhg`[RNKECBAA�� )-)'�����@BLNPWZSNBA=@@@@@@@@#,08<@<0(#�#&06;0#
���;@Ibw{|{{���{b\LI=;������������������������ ��������������FHTUWVVUUHECFFFFFFFFLO[chijh`[OGLLLLLLLLdm������������i_]]bd��
!
���!##/1;<@<82/##"!!!!!����������������������������������������7<HSQHG<967777777777�#�������#�-�/�<�?�<�;�9�/�#�#�#�#�=�<�5�=�I�V�Y�Y�V�I�=�=�=�=�=�=�=�=�=�=�H�>�?�H�I�U�`�a�g�d�a�U�H�H�H�H�H�H�H�H��غ׺��������������������!�����!�&�-�:�F�I�M�F�?�:�-�!�!�!�!������|��������������������������������s�p�g�f�f�g�h�s�t��������~�s�s�s�s�s�s�����������������������ʾ̾оʾ��������������������������ʾ̾Ͼʾƾ��������������M�E�A�K�W�f�s�������������������s�Z�M���	�����������	���������������r�c�[�K�D�G�\�����������������������[�B�5�(����)�5�B�[�t�}�t�g�[���y�`�Q�G�G�T�m�y���Ŀѿۿ������ڿ����;�.�%����������	��.�;�E�I�H�J�G�;����������������������������������)�(�)�*�1�6�B�G�C�B�?�6�)�)�)�)�)�)�)�)�T�Q�O�P�P�T�Y�a�f�m�r�z�|�{�z�r�m�a�T�T���`�G�.�"��	��#�G�q�y�������Ŀȿ��������ܹ̹ŹŹϹڹܹ���������������B�>�6�1�/�5�6�B�O�Y�[�h�i�n�h�e�[�O�B�B������~�������v�����������������������������(�5�A�G�A�<�5�(���������(�#��#�(�3�4�A�M�Y�Z�\�Z�R�M�M�A�4�(�(�g�b�Z�W�Z�[�g�s���������������������s�g�A�9�5�5�5�<�A�N�Z�a�`�[�Z�N�A�A�A�A�A�A��������������	�������	�����ŠśŔőŔŖŠšŭŶŹ������������ŹŭŠ�l�i�f�l�x�������~�x�l�l�l�l�l�l�l�l�l�là×ÓÑÈÂÂÇÍÓàìùý����þùìà�:�/�2�6�;�@�T�m�z�~�����z�m�c�T�S�P�H�:ĳĭĮıĿ�������
��#�+�#��
������Ŀĳ�������� ����(�,�5�>�A�J�A�3�(���l�l�g�f�i�l�u�x���������������������x�l������r�k�f�r�����������������������������Ƹƹƿ������������%�$����������̾f�`�]�f�k�s�����������ľ�����������s�f������ �����)�,�6�8�6�6�)� ����ֺкɺ����պ����-�:�D�P�U�J�F�!���ùù��������ùϹйڹϹùùùùùùùùú�����'�*�3�4�4�3�'���������3�3�:�=�@�C�L�Y�e�r���������~�r�e�Y�@�3�������������ĿͿѿҿѿĿ��������������������������������*�8�9�/�*���������׾վ׾�����������������.�"�"��!�"�.�4�;�?�;�5�.�.�.�.�.�.�.�.�������������ÿĿѿؿݿ��ݿۿѿĿ������������(�4�7�?�4�-�(�������ŹŭŠřŕŘŠťŭŹ������������������Ź����ùíÇ�v�o�zÇà�������������������������������������������������������������<�9�/�(�)�/�2�<�H�I�U�\�a�d�a�U�H�@�<�<�M�L�>�4�3�4�;�@�M�Y�a�f�q�r�w�s�r�f�Y�M�������*�6�=�7�6�*���������!� ��!�%�-�3�:�?�E�:�2�-�#�!�!�!�!�!�!�ܻû����������������ʻлܻ�޻ݻ���ܻ�л»��ûȻлܻ���'�4�9�.�'��������/�)�&�/�<�B�H�L�H�<�/�/�/�/�/�/�/�/�/�/D�D�D�D�D�EEEEE	ED�D�D�D�D�D�D�D�D��������������ùϹй׹Ϲù����������������Ľ½��������ĽɽнսӽнĽĽĽĽĽĽĽ�¦¦²¿������������������¿²¦E*E&E!E*E3ECEPE\EiEuEvEuEiEgE\EPECE7E/E*E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���������$�&�$��������������������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� 0 F * 7 4 & ` P 0 F � D : 3 _ L p I Z d ( u Q 7 9 0 5 4 8 K a l 7 _ 1 " N  W ) ) 4 E 5 i & . D  Q  . F F y _ P Q ^ k \ & g L g % D    �  d  �  l  �  s  �  �  �  $  �  �  �  U  �  J  �  1  i  =  &  *  �  ,  k  �  V  G  '  �  �  �  V  �  �  �  �  �    8  h  �  l  �  2  K  +  �  �  �    �  e  o  �  p  �    n  J  G  �  M  �  `  >  D  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  +       
  �  �  �  �  �  �  �  �  �  �  �  x  f  T  B  /       �  �  �  �  �  �  �  n  Z  E  1      �  �  �  �  �  q  �  �  (  N  o  �  �  �  �  �  n  Q  -    �  �  N  �  �    y  t  o  j  d  ^  Y  S  M  F  >  7  /  (       �  �  �  �  �  �  �  �  �  w  m  b  V  G  5    �  �  �  s  O  )    �  �    0  M  d  w  �  �  t  Y  2    �  �  \  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  i  ]  Q  E  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  a  =    $      �  �    .  +  &        �  �  �  t  B     �   `  �  �  �  �  x  p  b  I  &  �  �  �  g  .  �  �  �  A  �  �  �  �  �  �  �  �  �  �  �  �  �  o  Y  D  .    
  �  �  �  �  �  �  �  �  v  O    �  �  %  �  G  �  L  �  U  �  -  T  �  �  �             �  �  �  L  �  �  H    �  P  �  b  ,  U  P  I  =  1  &    �  �  �  �  Y    �  �  *  �  `   �  �  �  �  �  �  �  |  Z  ;  1  A  9  $  �  �  �  :  �  z   �                        �  �  �  �  �  �  �  �  �  M  R  V  <    �  �  �  �  x  V  1    �  �  �  s  k  �  �  u  �  �  �  �  �  �  �  �  �  s  _  H  0    �  �  �  �  �  �  �  �  �  �  z  `  A       �  �  �  �  �  x  ]  <   �   �     �  �  �  �  �  �  �  �  �  �  �  }  b  H  )    �  �  ~  O  M  N  M  N  O  S  W  V  O  :    �  �  z  B    �  �  �  k  ~    x  f  N  3    )  "  "      �  �  �  {  3  �   �       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  r  i  K  2    �  �  �  �  �    [  .    �  �  �  y  B    �  >  X  �  Y  �  �    '  G  \  W  H  .    �  O  �  �      �        !  !               �  �  �  �  �  �  z  [  <    /  D  S  ]  b  _  W  K  9  #    �  �  h  -  �  �  Q  �  W  d  o  w  {  y  q  c  Q  :    �  �    :  �  �  l  P  &  Y  Q  H  @  8  -      �  �  �  �  ~  _  A     �  �  �  �  C  ^  p  x  r  \  ?    �  �  �  o  >    �  �    _  �   �  M  F  ?  2  %    �  �  �  �  �  �  �  �  �  �  �  �  ~  Z    �  �  �  �  �  �  �  m  X  B  *    �  �  �  Z    �   �  �  �  �  �  �  �  }  i  T  ?  )    �  �  �  �  �  �  �  f              �  �  �  �  �  �  i  ?  %      �  $  �  �  �  �  �  �  �  �  �  t  b  O  F  J  *    �  �  �  O    �  �  �  )    �  �  �  {  X  B  D  >  %  �  �  �  w  1  �  �  �  �  �  �  �  �  �  �  �  n  V  ?  $    �  �  �  p  M  �  �  �  �  �  s  S  1    �  �  �  �  �  g  /  �  �  <  �  w  �  �  �  �  �  �  p  C    �  �  �  �  f  /  �  +  L  _       Y  �  �  �  {  i  S  9    �  �  �  �  h  1  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  \  :    �  �        �  �  �  �  _  *  �  �  �  R    �  �  i  (  �  �  N  L  J  G  E  C  A  ?  =  :  ;  =  @  B  E  G  J  M  O  R  �  �  �  �  �  �  w  c  M  3    �  �  �  �  �  D  �  �  g      �  �  �  �  �  �  �  �  �  p  ^  L  :  (       �   �  �  �  w  m  \  I  6      �  �  �  �  �  b  D  %     �   �  �  �      2  8  9  <  =  :  /    �  �  �  9  �  �    �  ;  :  :  8  5  1  '      �  �  �  �  q  N  +    �  �  �  ,  :  B  C  =  /    
  �  �  �  �  z  [  9    �  �  \    �  �  �  �  �  �    e  H  %    �  �  �  O  :  -  �  �  a  �  �  �  �  �  �  �  �  |  m  \  K  7       �  �  �  V    �  �  �  �  �  �  �  �  �  r  a  M  6      �  �  �  �  �  �  �  �  v  [  =    �  �  �  �  U  "  �  �  �  r  D  (  u  2  -  (  #        �  �  �  �  �  �  �  �  w  f  T  B  0  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  <  -         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  [  N  @  1  *  (      �  �  �  �  �  �  �  U     �   |  �  �  �  �  �  �  �  �  �  �  �  t  d  R  A  0      �  �  �  u  Z  @  )    �  �  �  �  S     �  �  {  ?    �  �  u  �  �        �  �  �  �  k  C    �  �  �  ^    �  �  S  J  B  :  1  )  !    
  �  �  �  �  �  �  �  v  a  M  8  $  Q  N  D  0    �  �  �  �  �  ~  j  O  +  �  �  m  /  �  ]  	P  	}  	d  	D  	.  	  	  �  �  �  X  %  �  �  z  E    �  [  S  �  �  �  �  �  �  �  \  !  �  �  K  �  �  M  �  �  ?  �  }  �  �  �  �  �  �  �  �  �  =      �  �  �  �  ^  )  �  �  �  �  �  �  �  �  �  �  u  _  F  (  
  �  �  �  �  U     �  �  �  �  �  �  ~  i  S  =  '    �  �  �  �  �  �  �  �  �