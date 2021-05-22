CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�E����       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Nz�   max       P��:       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��S�   max       <49X       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\)   max       @F�Q�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @v=�Q�     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @P`           �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @��            6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �n�   max       ;ě�       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�	   max       B0%�       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��<   max       B0>T       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       <�8K   max       C���       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ><�J   max       C��,       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          X       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Nz�   max       Px	       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�&���   max       ?�:��S&       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��S�   max       <49X       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?333333   max       @F�Q�     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @v=�Q�     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P`           �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @���           Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Aa   max         Aa       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�쿱[W?   max       ?�8�4֡b     @  [L         V      .            W   S   
                                 "         	   )   %      Q                     	   	                  A      #                        
               <            1   (      :         
N,6N�2zP��:OjݜP��N���N�]SOJ6�P��_P4=O,O��O�O��!N�Z�O���N��N�hoN�J�N8`O]��P��N���Nȼ�O�7O��5O�93OgSP2�Nی�O�6N�;�O�'�O�}O>�JN�-�N��NG$�NE>�N�XO?uO��kP�\�NE��O��N�S�N�
HN��O2N�v O�oN���O�wNZ4OX9�N8�O,<�O��N��6O'��N�w�P%6�O_�Nz�O�;-O�d�N��aN���<49X<#�
<o<o�D���D����o���
���
���
�ě��ě��o�t��t��D���D���T����1��9X�ě��ě��ě��ě����ͼ�`B��`B�����C��\)�t��#�
�#�
�,1�49X�49X�49X�@��D���D���D���D���D���D���L�ͽP�`�]/�]/�]/�aG��ixսixսm�h�m�h�m�h�u�u�����O߽�����P���㽛�㽰 Ž�E��\��S�`hot���~wth\````````nz���������|zpnnnnnn����/CHG8/��������W]`anz���������znaUW)5B[j���wNB5'$# %*16@CFHGDC6*({�������������{{{{{{ALSU[fgt���trpi[NC=Aku����������������nk�����+37::6)����emz�������zumhcceeeeS[agrt�����}tgg[UNPS�������������������������	
��������)16B76+).28;HafrrvqbaMH;6/,._anz����znnfa______��������������������{��������~wv{{{{{{{{15ABENOPNB>540111111PTafmovz}zumaTRKHIMPz�����������������}z�#()*)&���"#+///6759:3/#!""��������������������|���������������{uw|��������������������bmuzz���zwmmeeedhib2@O[h��������[B�������������������������������������������������:<ITblw{������{UI<3:��������������������������	���������������������������vz~������������zyuvv<BHOO[\^[SOHB<<<<<<<#)+(#Z[gtw���tg[[ZZZZZZZZ����������������������������
���������%1HPRbz���������aU4%���������������������������
	
����������������������������itw�����������}tiiii&)*6BKEB6*)(&&&&&&&&nv{���������{pnhgilny{�������������~{zwy��������������������ehjtu{|ytphabaaceeez{������������~{xxuz>BHNVUTNNB=;>>>>>>>>fjikt�����������tgf����������������������������������������qz��������������zrnq��������������������"#2<HIUY[WUN?</&#"#/<<@?<<0/.#��)BMNKPRPE5)���,09<BIKUY[YUMI<2-**,8<IKKKI=<58888888888&/HUdpx���zaUH</,&&��������������������otu}��������tkimoooottw{����������trtttt�<�7�/�)�.�/�1�<�H�K�J�H�<�<�<�<�<�<�<�<�������������ʼּּܼԼʼȼ��������������L�A�5�(�1�[�s�����������������������g�L�������~�s�m�s�x�������������������������H�>�;�?�R�a�����������������������z�a�H���	�����������	��"�"�%�"�"������������������������������������������޿y�m�`�T�M�G�E�T�_�`�m�w�y�������������y���л��x�l�:�.�'�:�S�����ܻ��������f�M��������4�@�Y�r���������|�x�r�f����������������#�#��������������M�K�M�U�Z�[�\�]�f�s�~������~�s�f�Z�M���y�m�T�G�@�;�6�4�;�G�T�Y�`�m�y���������M�<�4�/�(�4�A�M�f�s������������s�f�Z�M�ֺԺɺĺźɺͺֺ���������ֺֺֺ��s�g�Z�Y�Z�W�Z�g�s���������������������s��������(�)�/�(�(����������¿¿´¿�����������������������������˽����������������������������������������(�"�(�)�(�&�(�5�8�A�A�B�A�5�(�(�(�(�(�(ƎƃƇƎƗƚƧƳ������������������ƳƚƎ������������������$�0�:�>�>�;�0�'�����ŔŉŇŇŇŋŔŠŤŭŲŹ��żŹŭũŠŔŔ�/�.�#����#�/�<�H�U�a�e�c�a�U�H�<�/�/������������s�h�r������������ʾҾԾʾ��x�r�k�l�c�c�l�x�����ûܻ�ܻû��������x������2�@�I�X�[�c�f�x�x�k�Y�@���������������������)�)�1�)����������ҹ�ܹù����������Ϲܹ������������������������������*�+�*�(�������ŠŔŇ�{�p�{ŁŇőŔŠŭųŹżŽŽŹŭŠ�׾ӾʾǾʾ;׾������������׾׾׾������c�N�>�5�(�#�(�5�g�s�����������������ѿ������{�}���������ĿͿ����������àÓÇÁ�z�q�w�zÁÇÓÜæìù����ùìà��������������������������������������ؾ����������������ʾ׾����׾;ʾ������T�L�H�;�;�0�;�H�K�T�Y�a�k�a�T�T�T�T�T�T�B�B�<�B�O�[�`�g�[�O�B�B�B�B�B�B�B�B�B�B������������	����	�	������������������������۾ھ����	�������	�����a�\�Y�[�`�m�x�����������������������z�a���������m�H�8�:�J�m�z�����������������������������������ºƺ��������������������ɺƺ��������������ɺ���������ֺͺ��/�-�/�2�;�H�T�]�Z�T�H�;�/�/�/�/�/�/�/�/��üù÷íùù��������������������������ìììôöù����������ùìììììììì�~�����������������ʼ̼μʼɼ��������~�'�$����&�'�4�@�M�W�Q�Y�]�Y�M�A�@�4�'čċā�u�~āčęĦĳĺĿ����ĿĳİĦĚč��������������������	��	���������������S�R�O�Q�S�_�l�x�������������������x�_�S���������*�*�.�*�������������ŹŭŘŇŀ�|ŇŖŠŭŹ�����������������������������������������������������b�a�U�J�I�=�;�I�V�b�o�{�~ǅǈǈǂ�{�o�b�������������Ŀѿݿ�����������ܿѿſ��������¼ʼּ޼������������޼ּʼ��������������������������ĿѿӿӿѿϿĿ���D{D{DuDuD{D{D�D�D�D�D�D�D�D�D�D�D�D�D{D{àÇ��f�]�P�U�mÇÓìù������������ìà������ݽԽݽ���(�4�A�F�J�A�4�(�������������������������������������������k�e�q���������Ϲܹ���������߹ɹ����x�k����ľĿĺ����������&�3�4�0�#�����������
��#�&�/�9�<�H�J�H�G�B�<�/�#����²±¦ ¦²³¿����¿²²²² Y L 6 6 B ( X J c / 5 X L * 5 0 Q ? 1 E B . 8 k ~ W g � [ @ 9 1 S n b + 2 ~ B ~ > I 8 [ 5 ; f k < E : d J V A 4 [ 4 x A ! ? I E k C _ b  ]  �  �  �  �  �    �  �    5  Q    k  �  v  �  �  �  R  �  I     -  �    �  �  �    L  �  ,  2  �  �    �  P  �  K  �    �  ,  �  �  r  W  �  P  %  E  d  �  K  �    �  ~    �  �  0  �  �  �  �;ě���`B���-�D���H�9�49X�o��o��j��-��o�D���D������/�����ͼ�h������/�#�
�ixս#�
��w�+��C�����8Q��`B�L�ͽL�ͽD������e`B�ixսT���T���P�`�T���P�`��+�����`B�]/��e`B��%�ixս�+�m�h��hs�����7L�y�#���-�����hs��F��t���j������#�������n�������`����BWB��B��BsJBʇB0%�BY�B�<BY:B�[A�i�B	�gB!J6B�B�A�	B�aB�=B�B��A�f�B}B��B'�B��B UpB �}A��B�B�B��Bp&B'� B*��B�B��B�$B�B�B	|�B!�B�B��B ��B#H�Bf�B
�@B�{B(˔B)pQB�B�
B)GB!�B
S�B<sB��B�mB-7AB��B�B��B&|5B&��B��B�B
QB
�aB?�B�B@�B�6BV.B0>TB��B�FB�&BJ`A�W�B	 �B!IPB"MB�|A��<B�'B��B:�B��A�%-B>�B��B�B�#B 7�B ��A��B=�B7�B�\B=�B(WB*H"B@&B��B� BGBCRB	�0B>cB?�B}�B ��B#K�B|�B
��B�|B(��B)x
B=�B�UB)B�B�.B
FJB<�B��B�|B->�B��B�B@XB&�CB&�WB�hBÖB	�vB
>sA�nG@���A�ȊA�kA�ۙA[YDA��Ak�*@�77@��A�^A@��Ah�A@�.@?hA��dA4u*A���A!p5A�B�BO\B��A���A�AJ�@���@΋LA�t>��:A�"3A�^oAU!A���AzMZA�R�A�&AO��A��A��AZ3�AX�A��MA�e@!޷@<�A�'GA�`�A�A3@�=�@��A���A��l@�`�A��A���BҟB1fA{�.A�fAvc�C���A˲�A4cA"�<�8KA���A�]SA��A�A@�hA�L�A�p�A�\AZ�A�i�Al\�@� @я=A�YMA@ޘAg��AB	@>��A��A3�A�V�A �6A��(B]B	IYA�{A��AEAM@��@��A�t=>�I�A�t�A��>ATI�A�U�Az� Aʁ�A�W/AO+6A��A�\VA[qAYA��A��@+�4@4�7A�k`A�r-A�XH@���@�"�A��A���@��RA�wgA���B��B�-A{�1AVsAv��C��,A�`�A56�A!��><�JA���A�A���         V      /   	         X   S                                    "         	   )   &      R                     	   	                  A      #                     	                  <            1   (      :                  ;      +            =   -                                    '            %   '      1            %   '                           9      !                                                   -         '   #               %      %            ;   !                                    #                                 %   '                                                                                    -         '   %      N,6N�2zPM�O4Z|O��N���N�]SOJ6�Px	O�-�O,O��N��O^�N�Z�OFAN��N�hoN�J�N8`O]��O��TN���N��O�7O��*O] BOgSO��N��N��N�;�O�'�O�}O>�JN�-�N��NG$�NE>�N�XO?uO�XQO�k�NE��Ob�N�S�N���N��O2N�v O�oN���O�wNZ4O#0�N8�O,<�OFTN��6O�N�w�P%6�O?�ZNz�O�;-O�.N��aN���  �  h  �  S    O    l  +  m  �    v  �  �  i  I  b  4  N  �  )  6  �  �  �  �  �  	�  #  R  g  �  �     u  �  �  �  �  ~  p  �  w  �  o  �  �  s  �  �  A  �  �  �  �  �  
�  �  �  �  4  �  p  	�  Z  m  D<49X<#�
����;ě��o�D����o���
�#�
��/�ě��ě��t��e`B�t���C��D���T����1��9X�ě������ě����ͼ��ͽ�P��P��󶽉7L�t���w�t��#�
�#�
�,1�49X�49X�49X�@��D���D���L�ͽ��w�D���P�`�L�ͽT���]/�]/�]/�aG��ixսixսm�h�y�#�m�h�u��t������\)������P���
���㽰 Ž�Q�\��S�`hot���~wth\````````nz���������|zpnnnnnn�����
 ()%
�������Zbenz}����������znaZ )5B[f}wk[LB5)#('  %*16@CFHGDC6*({�������������{{{{{{ALSU[fgt���trpi[NC=A����������������zos�����!).0-)����emz�������zumhcceeeeS[agrt�����}tgg[UNPS��������������������������

������)16B76+)9;HTafhkg_TPHD;;4039_anz����znnfa______��������������������{��������~wv{{{{{{{{15ABENOPNB>540111111PTafmovz}zumaTRKHIMP�������������������#()*)&���##,/464893/#"####��������������������~�����������������|~��������������������bmuzz���zwmmeeedhib)06:DOX[hsvvxth[O6))�������������������������� ������������������������:<ITblw{������{UI<3:��������������������������	���������������������������vz~������������zyuvv<BHOO[\^[SOHB<<<<<<<#)+(#Z[gtw���tg[[ZZZZZZZZ���������������������������

����������`djnz����������znc``�������������������������

���������������������������qt}�����������~tqqqq&)*6BKEB6*)(&&&&&&&&nv{���������{pnhgilny{�������������~{zwy��������������������ehjtu{|ytphabaaceeez{������������~{xxuz>BHNVUTNNB=;>>>>>>>>tz�����������{tonlmt����������������������������������������z����������������{zz��������������������"#/3<HUYZVUM></)#"#/<<@?<<0/.#��)BMNKPRPE5)���.04<IUWYYXSLI<50,+,.8<IKKKI=<58888888888&/HUdpx���zaUH</,&&��������������������otu}��������tkimoooottw{����������trtttt�<�7�/�)�.�/�1�<�H�K�J�H�<�<�<�<�<�<�<�<�������������ʼּּܼԼʼȼ��������������s�g�X�J�G�H�N�g�����������������������s�������x�r�s�w�|�������������������������H�A�?�C�V�a�����������������������z�a�H���	�����������	��"�"�%�"�"������������������������������������������޿y�m�`�T�M�G�E�T�_�`�m�w�y�������������y���x�_�F�3�*�+�:�S�����л޻������ܻ��M�4�������4�M�f�r�w�}�~�{�n�f�Y�M����������������#�#��������������M�K�M�U�Z�[�\�]�f�s�~������~�s�f�Z�M�G�F�;�9�6�;�G�O�T�V�`�e�m�y����y�m�T�G�Z�W�M�A�6�4�A�M�Q�f�s������������}�s�Z�ֺԺɺĺźɺͺֺ���������ֺֺֺ��g�c�a�`�a�g�s�����������������������s�g��������(�)�/�(�(����������¿¿´¿�����������������������������˽����������������������������������������(�"�(�)�(�&�(�5�8�A�A�B�A�5�(�(�(�(�(�(ƎƃƇƎƗƚƧƳ������������������ƳƚƎ����������������$�0�8�=�=�:�0�%�������ŔŉŇŇŇŋŔŠŤŭŲŹ��żŹŭũŠŔŔ�/�/�#��#�/�<�H�U�a�d�c�a�U�H�<�/�/�/�/������������s�h�r������������ʾҾԾʾ����x�r�q�m�h�h�i�l�x����������������������������'�>�M�Y�f�p�p�f�Y�M�@�'�������������������)�)�1�)����������ҹ�ֹܹϹù��������ùϹܹ��� ����������������������)�&���������������ŇńŇŇŔŖŠŭŮŸŶŭŠŔŇŇŇŇŇŇ�׾ӾʾǾʾ;׾������������׾׾׾������c�N�>�5�(�#�(�5�g�s�����������������ѿ������{�}���������ĿͿ����������àÓÇÁ�z�q�w�zÁÇÓÜæìù����ùìà��������������������������������������ؾ����������������ʾ׾����׾;ʾ������T�L�H�;�;�0�;�H�K�T�Y�a�k�a�T�T�T�T�T�T�B�B�<�B�O�[�`�g�[�O�B�B�B�B�B�B�B�B�B�B������������	����	�	������������������������۾ھ����	�������	�����^�[�\�b�m�{�����������������������z�m�^�m�a�T�O�H�I�Q�T�a�m�z���������������z�m�����������������ºƺ��������������������ɺ��������������ɺֺ����������ֺ��/�-�/�2�;�H�T�]�Z�T�H�;�/�/�/�/�/�/�/�/����ùøïùú��������������������������ìììôöù����������ùìììììììì�~�����������������ʼ̼μʼɼ��������~�'�$����&�'�4�@�M�W�Q�Y�]�Y�M�A�@�4�'čċā�u�~āčęĦĳĺĿ����ĿĳİĦĚč��������������������	��	���������������S�R�O�Q�S�_�l�x�������������������x�_�S���������*�*�.�*���������ŝŔŉňŔşŠŭŹſ��������������Źŭŝ���������������������������������������b�a�U�J�I�=�;�I�V�b�o�{�~ǅǈǈǂ�{�o�b�Ŀ����������Ŀѿݿ�����������ݿѿȿļ������¼ʼּ޼������������޼ּʼ��������������������������ĿѿҿҿѿοĿ���D{D{DuDuD{D{D�D�D�D�D�D�D�D�D�D�D�D�D{D{àÇ��f�]�P�U�mÇÓìù������������ìà�������������(�4�A�D�H�A�8�(��������������������������������������������k�e�q���������Ϲܹ���������߹ɹ����x�k��������Ľ����������$�2�3�0�#��
��������
��#�&�/�9�<�H�J�H�G�B�<�/�#����²±¦ ¦²³¿����¿²²²² Y L 5 6 = ( X J b & 5 X H 4 5  Q ? 1 E B + 8 ^ ~ P h � 6 3 $ 1 S n b + 2 ~ B ~ > B  [ . ; S k < E : d J V 2 4 [ + x C ! ? @ E k J _ b  ]  �  �  �  L  �    �  �  �  5  Q  "  �  �  �  �  �  �  R  �       �  �  0    �    �  �  �  ,  2  �  �    �  P  �  K  T    �  �  �  �  r  W  �  P  %  E  d  i  K  �  �  �  f    �  �  0  �  w  �  �  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  Aa  �  �  �  �  �  w  i  [  J  5  !    �  �  �  u  L  #   �   �  h  ^  U  L  D  :  /       �  �  �  x  B    �  q  %  �  �  �    _  �  �  ;  m  �  �  v  E  �  �  g  V  !  �    �  >  ;  D  L  Q  S  R  P  M  H  B  7  (    �  �  �  �  v  7   �  �        �  �  �  �  �  r  h  S  ;  g  S    �  E  �  �  O  M  J  D  =  5  )      �  �  �  �  �  �  �  �  p  F                    �  �  �  �  �  �  �  �  �  �  �  �  l  f  _  V  L  A  2  #    �  �  �  �  �  �  k  D    �  �    +  (    �  �  �  `  �  �  �  �  �  t    ~  �  �  �   �  �  �    K  d  m  i  b  L    �  �  z  &  �    _  �  �  �  �  �  �  �  �  �  �  k  Q  7    �  �  �  �  h  D  !  �            �  �  �  �  �  �  �  �  �  �  �  �  w  z  }  �  d  h  k  o  s  v  t  s  r  p  n  l  i  f  c  X  J  ;  -    �  �  �  �  �  �  �  �  �  �  |  e  N  6    �  �    =  �  �  �  �  �  �  |  i  L  *    �  �  �  P    �  �  �  �  �  0  G  T  ]  e  i  g  `  T  E  -    �  �  �  o  -  �  V   �  I  ;  (    �  �  �  �  b  ;    �  �  p  .    I    �  �  b  N  7    �  �  �  �  h  F     �  �  �  K    �  �    �  4  ,  #        �  �  �  �  �  �  �  �  �  v  J     �   �  N  A  3  &       �  �  �  �  v  U  ;  %     �   �   �   �   �  �  �  �  �  �  �  d  H  /    �  �  �  �  w  V  4    �  �    )  $      �  �  �  �  �  }  Y  .  �  �  y  3  �  M  3  6  2  ,  #      �  �  �  �  �  �  p  O  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  J     �  �  S    �  �  z  s  f  V  F  3  !    �  �  �  �  �  �  �  �  �  �  �  d  y  �  �  �  �  �  �  �  h  -  �  �  f    �  5  �  ?  �  %  y  �  �  �  �  �  �  k  B    �  �  O  �  u  =  !  �  �  �  �  �  j  N  4  L  �  �  �  �  r  ^  G  0      �  �  �  �  �  g  +  �  	Y  	�  	�  	�  	�  	�  	Y  �  �  �  +    �  i  6  �      #        �  �  �  �  t  K    �  �  O     �  e  9  6  >  F  K  Q  D  1    �  �  �  �  ]    �  �  Q    �  g  _  T  F  1      �  �  �  �  �  Y  .    �  �  p  8  �  �  �  �  �  �  �  �  �  �  �  j  H  #  �  �  �  N  �  �  a  �  �  �  �  h  H  (    �  �  �  m  C    �  �  �  C     �         �  �  �  �  �  p  H    �  �  �  �  ~  u  c  I  +  u  f  X  E  0      �  �  �  �  �  u  b  O  @  3  &      �  �  �  �  �  �  �  �  w  l  ]  I  5       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �          '  0  9  B  �  y  p  g  ]  S  H  >  4  *           �  �  �  �  y  N  �  �  r  d  V  H  :  .  "        �  �  �  �  �  �  �  �  ~  {  v  m  ^  J  0    �  �  �  �  h  :  �  �     �  K   �  M  g  p  n  j  `  K  1    �  �  �  Y    �  �  B  �  E  r  a  L  L  X  H    �       �  �  g  %  �  v    �  �  F  �  w  i  Z  K  <  -        �  �  �    X  B  ,       �  �  a  �  �  u  k  a  ^  ]  [  T  B  -    �  �  m     �  �  }  o  g  ^  U  K  A  6  +       	  �  �  �  �  �  �  �  �  l  �  �  �  �  �  p  W  @  *    �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  |  o  b  U  I  <  /  !       �  �  �  �  s  i  ^  P  @  .    �  �  �  �  �  [  !  �  �  P  ]  j  v  �  �  �  �  �  �  �    w  n  c  U  G  :  ,  )  (  (  (  (  �  �  �  }  c  D     �  �  �  p  C  	  �  �  A  �  �    g  A  :  2  +  %    	  �  �  �  �  �  �  �  �  �  ~  `  #  �  �  �  �  �  �  �  �  �  �  �  o  V  ;       �  �  �  �  y  �  �  �  �  �    z  s  k  d  \  T  M  H  G  F  E  D  D  C  l  �  �  �  �  �  �  {  h  Q  5    �  �  v  9  �  �    ?  �  |  f  P  <  (    �  �  �  �  w  ?    �  �  U     �   �  �  �  �  �  �  �  �  h  I  *    �  �  �  �  Q    �  �  ;  	�  
4  
\  
v  
�  
q  
O  
!  	�  	�  	d  	  �  D  �  %  {  �  �  �  �  ~  i  U  B  .  *  0  6  3  .  '    �  �  �  �  z  H    �  �  �  �  p  E  
  �  �  d    �  j    �  '  �  Q  �    �  �    ^  6    �  �  c  '  �  �  W    �  \  �  j  �  �  4    �  �  E  �  �  o  @  �  �  O    �  p  �  5  �  �  &  �  �  �  �  �  �  �  �  b  &  �  �  N  �  �  H  �      z  p  h  `  X  Q  I  A  9  1  )  !      
      �  �  �  �  	�  	�  	�  	t  	A  	  	  �  �  �  S    �  g  �  s  �  �    $  U  Z  I  5  !    
     �  �  �  �  [    �    2  �  c    m  X  D  .    �  �  �  |  O  #  �  �  �  �  {  q  L    �  D  5  &    	  �  �  �  �  x  N    �  �  o  8  	  �  �  