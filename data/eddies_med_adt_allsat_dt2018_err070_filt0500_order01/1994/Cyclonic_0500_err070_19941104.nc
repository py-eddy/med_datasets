CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����E�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       <ě�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @Fs33333     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v��\(��     	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @4�        max       @Q@           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��           �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��P   max       <T��      �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�wR   max       B1��      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��3   max       B0��      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�    max       C��#      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�E�   max       C���      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          a      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P���      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��a@N�   max       ?֞�u      �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       <�j      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @Fs33333     	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v��\(��     	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @Q@           |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���          �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�bM���   max       ?֞�u       T�                        .         
         >      	            \         
      *                        	         8               a         &         	                                          N�7�M�^aNe˺N�<Ni�KN8k�O���P!��Np�O���O��M�OjU�P���Ot3�NJN�O���O�nO?�P���N��kO8>�N�NQOʵNs�]N$�zN���N7��N(��N8�N��OAN*�O�6O�b�O]�O)�?N��N���Px;�N/��O��O��VN�gRN�{CN�H�N�b�OR�O=_(N��N[��N��O^�N��COB�M��N6nN��O^LNŤ<ě�<�C�;ě��o���
��`B�t��t��T���u�u��o��C���9X��j��j��j������`B��`B��`B��`B��h��h��h�����+�t��#�
�',1�0 Ž0 Ž0 Ž0 ŽH�9�L�ͽL�ͽL�ͽT���aG��ixսixսixսixս}󶽁%��%��%��+������P���㽰 Ž�-��-��Q�ȴ9�����<<IQU_bnopngbUIG><<<##-#rz���������zxprrrrrr./7<HJLLHH</.++,....��������������������pz}������zsopppppppp�����);@A6	���������������������������������������������3;Hamqz|{maTHEE?90/3��������������������BBOT[\[QONBBBBBBBBBB���������������������#0Bn{����nVL<0������"&!����������������������������
#08:33640#�����BBOS[hjqojjha[UOCBAB��������������������������
��������ux�����������������������������������������qtw�������tpikqqqqqq��������������������(.6CO\u�����uhO6+&&(<BNNO[dgog[NEB<<<<<<���������v����������.6ABJO[ehksh[OB960..#)16:<6) $##########����������������������
������������������������������������������")5BIB<5)""""""""""JNY[gt�������tng[QNJ��������������������#.<HUaghfaUHD</#ajmz|}��������~mb^^a����������������������������������������#/HVaokUT_n���z/##��������������������mmoz������zmlgebgm������������������������������������������������������������KNQ[dgoqg[NMKKKKKKKKIO[^hinopnhd[YOOHEIIsu�����������troqspsw|}�������������{vuwy{����������{zvtrqyy���������������������� �����������������%&������������������������������		 �����������##%%#"���������������������������������QUV_anz��������znaUQ&))15BMNSNNB<5))(%&&�������������������ʼ˼ּټݼؼּʼ�����E7E5E7ECECEPEXEPEPECE7E7E7E7E7E7E7E7E7E7��������������������������������������������������������������������������ҿm�l�m�n�r�y�������������y�n�m�m�m�m�m�m�Z�V�Z�e�g�s�������z�s�g�Z�Z�Z�Z�Z�Z�Z�Z�g�[�N�0�$�$�5�B�N�[�o�x¦¯ª�M�4�����4�M�v�������������������s�M�ݽ۽нĽ����Ľнֽݽ���������ݽݽݽ���������(�5�W�p�|�}�s�g�Z�A�(���������Ŀʿѿݿ�����������ݿѿ˿Ŀ����a�a�\�a�l�n�p�zÃ�z�o�n�a�a�a�a�a�a�a�a��ݿԿѿʿȿʿѿݿ���������������������v�n�g�M�H�P�s���������������������N�F�D�K�V�Z�g�s���������������z�s�g�Z�N��y�����������������������������ܻ������ɻлܻ�������$�9�K�@���Ϲι����������ùϹܹ�����������ܹ�ÇÀ�|ÁÅÍÓÓàìïôùøñìàÚÓÇ����������������I�{ŠŪ��{Œł�b�I������������������������������������������g�N�F�J�V�Z�g�s���������������������s�g�/�-�/�5�<�H�R�U�U�^�Z�U�H�<�/�/�/�/�/�/�U�M�H�D�E�H�U�V�[�V�U�U�U�U�U�U�U�U�U�U���־ʾžž̾׾���	�� �$�%���	�����V�P�I�I�I�N�V�Z�b�e�j�i�b�^�V�V�V�V�V�V�V�U�I�I�I�V�b�n�m�b�V�V�V�V�V�V�V�V�V�V���������������������û˻˻ʻǻû�������ùõùü������������ùùùùùùùùùù�C�B�B�C�O�\�f�\�Y�O�C�C�C�C�C�C�C�C�C�C�����������������������������������뻪�����������z�������������û̻û��������.�"��	������������	��"�$�.�0�8�1�.�	����	������	�	�	�	�	�	�	�	�	�	��������������	�������	���čā�t�j�[�O�E�C�O�[�hāĚĢĦīĦĚġč�M�G�A�:�7�8�A�M�Z�f�o�t�x�s�l�o�n�f�Z�M��ƳƱƩƳ���������������������������� �����������������������T�O�N�T�a�m�z���|�z�m�a�T�T�T�T�T�T�T�TE�E�E�E�E�E�E�E�E�FFFF$FJF`FRF:FE�E��~�|�z�y�~�������������������~�~�~�~�~�~ĚęĚĞĦĳļĿ����������������ĿĳĦĚĿļĳĬģęĖĖĚĳĿ����������������Ŀ�����߿ݿտݿ��������������������������������������ĿǿĿ���������������������������������������������������������������������������������������������������������	��"�&�.�/�.�(�"��������׼�����������4�@�M�Y�`�Q�M�@�4�'��S�R�P�S�_�k�l�m�x�������������x�l�_�S�SD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��0�,�.�0�=�I�I�L�I�=�0�0�0�0�0�0�0�0�0�0�r�l�l�r�}���������������������������r�ʼɼüʼʼּ��������ּʼʼʼʼʼʽ������!�+�.�:�E�G�M�G�:�6�.�!���ݽѽݽ�����������ݽݽݽݽݽݽݽݼ���������!�$�!�������������������������������&�%�������������������|�z�m�k�e�b�`�a�i�m�z����������������ìààÓÏÌÓÔàììóùþ������ùìì D y B 6 k X t Q t A W G @ ` M S S . L F 6 Y < � 3 Y ` W > K R i F f \ = I � u R p � ; 4 M 7 ) U D P F i F , 8 @ � n R ] P    �  *  �  �  �  y  I    �  �  �  /  �  n    w    :  �  �    �  �  �  �  �  /  D  Y  D  Q  �  D  g  �  �  �  )  L  �  	  �  L  |  �  �  �    �  �  �  �  D  R  �  I  3  M  �  q  �<T��<T��:�o�e`B�t��#�
��w�]/����,1���ͼ�j��/����C����@��P�`�H�9���C��<j���t���hs�\)�+�Y��P�`�8Q�49X�Y��P�`�8Q�m�h������hs�y�#�e`B�aG���P�ixս�\)�\��7L��o��\)���w��E����-����� Ž������`�Ƨ�Ƨ�E��ě������n�B'2�B�oB��B��B	BBTB��B �UB!�SA�wRBըB�@B�B&nlB:(B l�B$@�B8B��B;gBkB�3B�!B �gB1��Bl�B
��B��Ba�BWBjB��B�B"�B	��B�B��A��_B�B	B��B \�A��nB�B*D�B+��B�BC�B)�B)��B(��B�'B��B�B4�B��B$�JB.NB�{B�Bp�B'9%B��B}{B��BHBBD�BA�B >�B!��A��3B�XB� B-)B&B=B�mB �`B#��B?�B�OB<�B~ BQ�BN�B �B0��B<�B
N�BD�BL7B@B��B��B�YB%SB	��B�ZBBA�z�B�AB@B��B �|A��B?DB*<�B+�B��B>#B\IB)�	B)5�B�pB��B>mB?�B@)B$��B.> B��BBH@���C���Aп�A���An�A��0A�y�A@�tA*�?A��A}lgA�e�A�qA��A�]�@�̡@�Y>� A��LA�*'A���A��A�>6A�n~AX��BBG@�<"A��9BY�A���@���A\w�A[��AY7oA��mA>�JB��B��A���C��#@��A�bA�2�A�hAtQA�؁A��aA�}�@�>@���C�PB
��@�!�A�kAA�A-�[A	~]A��A���A�T�@��PC���A�{�A���Ao��A���A�k�AA�A* ;A���A}�A�~oA~��A�.�A���@��@���>�E�A�}�A�A���A�vAA�jBA��TAYcB��B?�@��sA�z�B�FA��$@��A]��A\AY�A܋OA= �B��B�WA�@�C���@�A�}mA�yA��lAs]�A��!A�y�A�\�@ͦ�@��zC�'�B
��@��A ��A NA,T=A	�A��JA��Á2                        .                  >      	            ]         
      +                        	         9               a         '         	                                                               %   /                  =         %         ?               !                                 #               A                                                                                 %   )                  =         %         -               !                                                A                                                            N�i{M�^aNe˺N��pN��N8k�O��jP<MNp�O���Nr�M�OjU�P���Ot3�NJN�O���N���O?�PFUN��kN�_�Ng^�NQOʵNs�]N$�zN���N7��N(��N8�N��"N��N*�OyzO;'�O]�OP*N��N���Px;�N/��O��O�rTN�gRN�{CN�H�N�b�OF�O=_(N��N[��N��O^�N��COB�M��N6nN��O^LNŤ  ,  )  )  �  /  �  I  *  1  �  C  #  �  �      )      	�  d  �  �  $  �  �  �  T  �  �  >  �  �  �  �  �  �  �  �  �  �  b  R    T    �  V     �  �  2  d  Q  �  J  1    v  b  �<�j<�C�;ě���o�ě���`B�#�
�T���T���u���㼃o��C��ě���j��j��j����`B�T����`B�C�����h��h�����+�t��#�
�'0 Ž49X�0 Ž49X�}�H�9�P�`�L�ͽL�ͽT���aG��ixս�%�ixսixս}󶽁%��o��%��+������P���㽰 Ž�-��-��Q�ȴ9�����>ITUXblmebUII?>>>>>>##-#rz���������zxprrrrrr./<FHJJH><8/-,......��������������������pz}������zsopppppppp�����):@@5��������������������������������������������3;Hamqz|{maTHEE?90/3��������������������BBOT[\[QONBBBBBBBBBB���������������������#0En{����{UK<0������"&!����������������������������
#08:33640#�����LOY[dhnmhhg[OJFELLLL�����������������������������������������������������������������������������ttu�����tqkltttttttt��������������������(.6CO\u�����uhO6+&&(<BNNO[dgog[NEB<<<<<<���������v����������.6ABJO[ehksh[OB960..#)16:<6) $##########����������������������
������������������������������������������")5BIB<5)""""""""""O[`gt�������xtqg[TOO��������������������#.<HUaghfaUHD</#_ablmz{|�����mca__����������������������������������������#/HVaokUT_n���z/##��������������������mmoz������zmlgebgm������������������������������������������������������������KNQ[dgoqg[NMKKKKKKKKIO[^hinopnhd[YOOHEIIrt������������trprurw|}�������������{vuwy{����������{zvtrqyy���������������������� �����������������%&������������������������������		 �����������##%%#"���������������������������������QUV_anz��������znaUQ&))15BMNSNNB<5))(%&&���������������ʼּؼۼ׼ּʼ�����������E7E5E7ECECEPEXEPEPECE7E7E7E7E7E7E7E7E7E7���������������������������������������������������������������������������߿y�s�u�y�������������y�y�y�y�y�y�y�y�y�y�Z�V�Z�e�g�s�������z�s�g�Z�Z�Z�Z�Z�Z�Z�Z�g�[�N�1�%�%�5�B�N�[�o�wª¥�����4�M�d����������������s�Z�M�4��ݽ۽нĽ����Ľнֽݽ���������ݽݽݽ���������(�5�W�p�|�}�s�g�Z�A�(���ݿԿѿ˿пѿݿ�����ݿݿݿݿݿݿݿ��a�a�\�a�l�n�p�zÃ�z�o�n�a�a�a�a�a�a�a�a��ݿԿѿʿȿʿѿݿ���������������������w�p�g�P�M�U�s���������������������N�F�D�K�V�Z�g�s���������������z�s�g�Z�N��y�����������������������������ܻ������ɻлܻ�������$�9�K�@���ù����������ùϹܹܹ�����ܹϹùùù�ÇÀ�|ÁÅÍÓÓàìïôùøñìàÚÓÇ�0��
�����������
��<�{Ń�p�n�r�q�b�I�0�����������������������������������������Z�P�S�Z�`�g�s�u���������s�g�Z�Z�Z�Z�Z�Z�/�/�/�6�<�H�U�]�Y�U�H�<�/�/�/�/�/�/�/�/�U�M�H�D�E�H�U�V�[�V�U�U�U�U�U�U�U�U�U�U���־ʾžž̾׾���	�� �$�%���	�����V�P�I�I�I�N�V�Z�b�e�j�i�b�^�V�V�V�V�V�V�V�U�I�I�I�V�b�n�m�b�V�V�V�V�V�V�V�V�V�V���������������������û˻˻ʻǻû�������ùõùü������������ùùùùùùùùùù�C�B�B�C�O�\�f�\�Y�O�C�C�C�C�C�C�C�C�C�C�����������������������������������뻪�����������������û˻û����������������	�����������	���"�-�.�6�.�"��	�	�	����	������	�	�	�	�	�	�	�	�	�	������������	��������	�����h�g�\�[�Q�V�[�h�tāĒĚěĜĚĕčā�t�h�M�G�A�:�7�8�A�M�Z�f�o�t�x�s�l�o�n�f�Z�M������ƳƲƮƳ�������������������������� �����������������������T�O�N�T�a�m�z���|�z�m�a�T�T�T�T�T�T�T�TE�E�E�E�E�E�E�E�E�FFFF$FJF`FRF:FE�E��~�|�z�y�~�������������������~�~�~�~�~�~ĚęĚĞĦĳļĿ����������������ĿĳĦĚ����ĿĲĦĞĜĦĳĿ�������������������̿����߿ݿտݿ��������������������������������������ĿǿĿ�����������������������������������������������������������������������������������������������������������	��"�%�-�.�-�&�"������������������4�@�M�Y�`�Q�M�@�4�'��S�R�P�S�_�k�l�m�x�������������x�l�_�S�SD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��0�,�.�0�=�I�I�L�I�=�0�0�0�0�0�0�0�0�0�0�r�l�l�r�}���������������������������r�ʼɼüʼʼּ��������ּʼʼʼʼʼʽ������!�+�.�:�E�G�M�G�:�6�.�!���ݽѽݽ�����������ݽݽݽݽݽݽݽݼ���������!�$�!�������������������������������&�%�������������������|�z�m�k�e�b�`�a�i�m�z����������������ìààÓÏÌÓÔàììóùþ������ùìì C y B + J X r Q t A $ G @ a M S S 3 L D 6 2 = � 3 Y ` W > K R ? T f X * I � u R p � ; ) M 7 ) U A P F i F , 8 @ � n R ] P    �  *  �  �  >  y    �  �  �  {  /  �  _    w    �  �  �    �  �  �  �  �  /  D  Y  D  Q  �     g  L  �  �  �  L  �  	  �  L    �  �  �    �  �  �  �  D  R  �  I  3  M  �  q  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  +  ,  +  &  !        �  �  �  �  �  �  �  i  D    �  �  )  '  &  $  #  !          �  �  x  9  �  �  �  �  �  j  )  "          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  W  5    �  �  �  o    �        $  *  ,  $      	  �  �  �  �  �  y  `  G  .    �  �  �  �  �  �  �  �  �  �  �  �  x  s  x  |  �  �  �  �  G  H  5    �  �  �  �  �  �  �  F  �  �  R  �  s  m  �  �    &  *  '        �  �  �  �  �  �  �  X    �  8  �  �  1  ,  '  #         �  �  �  �  �  Y  4    �  �  �  �  c  �  �  �  �  �  �  �  �  �  p  T  .    �  �  8  �  z    K    &  6  /  )  "  $  7  >  3  #    �  �  �  �  P     �   �  #         �  �  �  �  x  S  .    �  �  �  a  6  	   �   �  �    w  k  ^  P  A  1  "      �  �  �  �  �  z  g  F  %  �  �  �  `  *  �  �  E  �  }    �  F  �  �  s    �  B  �    ~  z  k  Z  I  4  !      �  �  �  �  �  w  S        �        �  �  �  �  �  �  �  �  �  �  |  l  \  M    �  _  )  #                      �  �  }  ?    �  �  �  �  �        �  �  �  �  �  �  �  �  p  ,  �  ~  �  J  �    �  �  �  �  v  L    �  �  n  -  �  �  <  �  c  �  I  �  <  �  	  	l  	�  	�  	�  	�  	X  	$  �  v  �  b  �         1  A  d  Z  O  D  8  ,         �  �  �  �  �  �  �  �  w  f  U  �  �  �  �  �  �  �  �  �  �  �  �  v  c  N  7      �  �  Y  q  �    p  a  R  B  2  !    �  �  �  �  �  �  k  M  /  $  6  H  W  b  l  n  j  f  f  f  g  k  o  v  ~  �  �  �  �  �  �  �  �  �  �  �  v  W  0     �  �  [    �  f  �  Y  e  �  �  �  �  �  �  �  �  u  c  Q  @  +    �  �  �  �  �  �  �  �  �  �  �  �  �  |  {  y  w  v  t  s  u  w  y  {  }    T  <  #  
  �  �  �  �  �  �  �  �  [  $  �  �  u  d  W  G  �  �  �  �  �  �      &  #        �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  y  c  J  0    �  �  �  �  >  8  2  +  %            �  �  �  �  �          &  �  �  �  �  �  �  �  �  �  o  I    �  �  �  J    �  �  \  �  �  �  �  �  �  �  �  �  �  }  f  O  6        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  e  L  4    �  x  �  F  }  �  �  U  �  �  w  X  /  �  �  A  �  #  i  �  �  �  �  �  �  �  �  �  j  B    �  �  �  �  �  L    �  �  e  =  �  �  �  {  b  K  E  @  "      �  �  �  n  >    �  �  r  �  �  �  �  �  �  ~  m  [  I  4      �  �  �  �  w  Y  :  �  �  }  {  x  u  r  o  l  j  g  e  Z  I  8  '  �  �  �  Y  �  g    �  �  ;  �  d  �  �  
�  
C  	�  	K  	�  	|  �  �  �  p  b  [  U  N  H  A  ;  4  .  (        �  �  �  �  �  �  �  R  7    	  �  �  �  �  �  �  j  F     �  �  s  �  �  M   �  �  �  �       �  �  �  �  `  2    �  �  ]    �  (  �  �  T  =  %  	  �  �  �  �  �  o  N  (  �  �  �  U    �  �  q    }  z  s  b  R  @  /    
  �  �  �  �  �  t  T  4     �  �  �  �  �  �  ~  m  ]  K  9  &      �  �  �  �  �  �  n  V  G  7  $    �  �  �  �  �  g  I  +    �  �  -  �    �  �  �  �  �  �  �  �  {  U  ,  �  �  �  V    �  E  �  �  �  �  �  �  �  �  {  g  Q  8    �  �  �  �  |  \  0  �  �  u  �  �  �  �  z  e  O  8       �  �  �  �  �  p  S  4     �  2  (    
  �  �  �  �  �  s  S  4    �  �  �  �  d  �  m  d  \  S  J  B  <  5  .  &        �  �  �  �  �  �  �  �  Q  K  @  0    �  �  �  m  8     �  �  d  "  �  $  �  �  H  �  �  �  �  �  r  _  O  D  -     �  �  b  *  �  �  x  9   �  J  -    �  �  �  �  �  �  r  Q  6  G  u  �  �  �  	  )  I  1  0  .  ,  +  )  (  &  %  #  !                         �  �  �  �  �  �  z  e  J  (    �  �  �  x  S  -    v  ]  M  J  '    �  �  H    �  t  ?  �  u    �  ,  �  C  b  O  :  %    �  �  �  �  �  c  C  )  
  �  �  l    �  O  �  �  t  ]  F  /        �  �  �  �  L  �  �  /  �  E  �