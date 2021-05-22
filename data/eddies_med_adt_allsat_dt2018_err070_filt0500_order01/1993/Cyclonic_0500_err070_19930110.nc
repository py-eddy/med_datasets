CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���Q�     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       PA�#     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��v�   max       <�j     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?}p��
>   max       @F7
=p��       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vt�����       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P`           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�@         4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       <�C�     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B/�6     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B/�$     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =:�\   max       C�zH     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�~   max       C�i     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          G     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       P��     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�Q   max       ?��"��`B     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��v�   max       <�j     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?}p��
>   max       @F&ffffg       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �У�
=p    max       @vt�����       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P`           �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�@         4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =c   max         =c     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Q�`   max       ?��"��`B       cx         "      C                  "            
               %                           7            
      F            .                                        +      	                           !            9                               N6�9N"eO,��N4e�O̞ NH��OSqQO*��N�TIN��Oo�rN���OH��O\avN6�)O��lN"v�O�N�#�O���N��N�O�sPO���N���OE�N۞N0	bO���OnN�ĄO:��Nq�_N
i�P7�vO&�uN��VO3��PA�#NNMbO���N�/(N���Oi�HO'�O�
>N�{YO$O ��O:~�O$>�O��|N�KFN��O$�DNא�OEE!O��EO/)N�,�Nm�mN�4O��N���O=7�N?�5Pn�N��Oũ�N.O�[mNK�N�UO�aN���NS�3N��X<�j<��
<�o;D��;o�D���D���D���D���D���D�����
���
�ě��o�#�
�#�
�49X�49X�49X�D���T���e`B��o��C���t���t���t����㼬1��1��9X��9X��9X��j�ě����ͼ�����h��h�����C���P��P��P��w��w�#�
�#�
�,1�,1�,1�,1�0 Ž0 Ž49X�49X�49X�49X�8Q�D���H�9�L�ͽ]/�aG��aG��aG��aG��aG���%��7L��C���\)��\)���㽾v�)*-*)�����������������������
#*-#
�������������~�������������
#)/0-,#
�������<?G</'##/0<<<<<<<<<�����
��������=BJN[_fgigee`[NB<66=MN[bdghige[ONIIEMMMMS[gt�������tkgb[XSS����

����������������������������lqty����������ytgdal)6COQ[WPJ;6*=BOX[b[VONB?========��������������������pt������trpppppppppp	)+5;A5/)��jnxz{������znonhjjjj������ ������������

 ���������!#$(&#��������������������swz�����������tnkjpsNOS[bhhjhfa][OHKNLNN#//:<DGC<2/##%-0<INQIB<:40--------��������������������kn|������������~zqikrz��������~zrnrrrrrr����������������������������������������������������������������������������������������������������������������������������

������%-BN[g��������g[B4*%��������������������������������������������������������������������������������������������pt���������������{qp�
-<INOK@30
 ��z�������������zzzzzz���������������������������	�����������������������������s}������������ztroos�����������������Y[htvvtttpha[VPQYYYYrt{��������yuttrrrrrSTahmoqqomjeaTTPOOQSQTabgjihgaTQPPMPQQQQ��������������������[gt��������tje^_[UT[�����������������������

 ������������������������������������������gqy�������������tn_g������������$)1669766)��������������������`gt����������l_[OOQ`NUXanosusnla]UQNNNNN�������������������������	

����������������������������������� ������������������������������������

������'0<IIKLII@<10$'''''' ��#'/4:730/##!�ɺɺ������Ⱥɺֺ����ֺɺɺɺɺɺɺ��t�g�tD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��нͽƽнݽ�����ݽннннннннн�ECE*EE E$E*ECEPE\EuE~EzE}E�E�E�EuEiE\EC�ݽ������������ݽݽݽݽݽݽݽݽݽݽݿ	�����ؾо׾����	��"�(�.�6�.�*��	�I�>�0�-�,�0�=�A�I�N�V�b�o�t�y�t�o�b�V�I�A�?�7�A�M�W�Z�f�m�s�y�s�q�f�Z�M�A�A�A�A����������������������������f�Z�M�J�O�Z�f�s��������������������s�f�y�t�m�`�_�U�`�m�x�y�����������������y�y����������������������������	���	�����	����߾Ҿʾʾ׾���	�"�(�0�3�3�.�"��a�[�]�a�e�n�y�z�}�z�q�n�a�a�a�a�a�a�a�a�ֿ��ÿѿݿ�����5�9�A�=�5�(��������H�?�=�H�R�U�V�^�Z�U�H�H�H�H�H�H�H�H�H�HĿĻĳĬĪĲĳĺĿ������������������ĿĿ�N�D�A�A�A�N�Z�Z�g�s�y�}�s�g�f�Z�N�N�N�NÓÉÅ�z�v�wÁÇÓàìùÿ����ÿùìàÓ�������������������ü����������������������������������������������������������߼r�f�Z�\�f�o�r�g�r��������������������r����s�f�\�\�e�s���������ʾ־ھ׾ʾ��������������������������ĽнӽսнĽ��������<�<�3�/�/�<�>�H�U�a�h�n�q�u�n�d�a�U�H�<�������������ĿſĿ����������������������M�J�F�F�M�Y�b�f�h�f�Y�O�M�M�M�M�M�M�M�M�F�:�2�4�8�:�A�S�l�����������|�x�l�_�S�F�(�%������(�4�A�M�W�Z�\�Y�M�J�A�4�(��޼޼��������������������������"�(�"��/�3�;�H�T�P�H�;�/�"���������������$�����������ù����ùϹعܹ߹ܹϹùùùùùùùùùúL�@�1�0�8�:�@�L�i�r�~�������ĺȺ����~�L�s�g�\�U�N�L�E�N�Z�g�s�y���������������s�U�S�S�U�Z�a�b�n�u�zÀÁ�|�{�z�n�a�Z�U�U������������������������������������������������������������������������������)�5�B�C�B�?�5�)���������#�"�0�6�:�<�G�U�b�{ŅŔŖŇ�n�b�L�<�0�#�y�w�y�����������ĿϿѿӿѿĿ����������y¦ª²´²¦��x�s�v�{�����������ʼѼɼ����������������x�x�����x�v�x�����������������������ܻû��������������ûлܻ���������čćā�y�t�j�l�tāĈčėĖĕčččččč�h�e�[�Z�O�P�O�M�O�[�h�tāĆċČĂā�t�h�������������������������������������������
���	��������'�4�@�@�:�4�'��������������������
�������
�������#��
�����������
� �*�<�O�U�l�b�U�I�<�#���������������������������������������������������������	���	�����������������Z�V�?�A�F�N�Z�g�s�����������������s�g�Z�����������������������������������������������������z�q�u�y�z�}�����������������������	�"�.�;�L�J�K�T�T�G�;�7�"�	���T�Q�M�G�A�;�.�,�)�.�;�G�T�`�k�j�d�`�W�T������������!�-�3�-�!� �����������4�+�4�5�@�C�M�U�V�W�M�@�4�4�4�4�4�4�4�4�������s�m�g�f�g�s������������������������ŭŠřŔŌōŗŧŭŹ�������������������B�7�6�*�+�6�B�O�U�[�]�[�U�O�B�B�B�B�B�B�Y�M�D�G�M�O�Z�f�s�����������������f�Y�O�K�O�R�U�[�[�[�^�h�l�h�d�[�O�O�O�O�O�O�����������
��/�H�a�h�f�]�H�/�!������ع����������������ùιϹѹϹɹù����������3�.�1�:�D�D�L�r���������������r�e�L�@�3�ɺź������������úɺʺ̺ɺɺɺɺɺɺɺɼ�{���������ʼּ��� ����ּ���������������$�&�0�4�8�0�0�$���������S�R�G�G�C�G�S�X�`�j�`�T�S�S�S�S�S�S�S�S�нϽĽ��������������Ľнݽ������ݽн������������ĽнѽннĽ½�������������ÇÇÇÌÓàìôõìàÓÇÇÇÇÇÇÇÇE�E�E�E�E�E�E�E�E�F
F E�E�E�E�E�E�E�E�E� > � p 4 < L ^ 3 B e D a ` o M W D 5 H R m O . D c  8 G \ 8 3 G T L  A b ? _ ? d ] C e W = ] C ! s - x ) k z q # + L = G Q 8 - J v I * c c o \ S , > M d    W  X  �  ;  �  d  �  o  �        �  E  c  �  _  a  �  q    :  �  �  �  I  $  c  !  �  �  �  l      {    �  �  z  �  Q  �    �  �  �  r  P  �  a  �  �  �  �  6  �  V  O  �  �  �  �  �  �  �  �  �  K  D  �  �  @  7  �  �  �<�C�<�o��C����
��o�o�ě���t��t����
��P�t���1���㼓t��#�
�u������D�����㼓t��t��#�
��j�,1���
��9X����49X�T���C�����`B��vɽ'�w��w�����+�P�`��P�8Q�q���P�`�q���@���7L�T����hs�q����1�]/�P�`��o�P�`��C���hs�L�ͽH�9�P�`�aG���m�h��t��u��`B���P�� Žm�h��{������P�� Ž��㽥�T����BӥB�wB �B)�~B�B_GB��BL�B��B	��BM%B=�B
TNB/�6B�RB�OB�B�DB8/B�B$BeB!,JBYWB�B B��B&O"B�[B39B�]B9B�	B�B�yBJ�B"C�B�B	��B�HB�B+'{B�B-zB��B%%�B
�B�]B�dB��B
�B�B��BńA���A��B-RB	�HB�IB#��B)�
B�AB1B1�B�dB�$B
aBI�B![�B#��B,[�B�qBN�B#��B&!Bl�B0�B�B�YB�lB)�:B1�BB�B�B@B�B
 �B?6B@HB
AB/�$B��BFXB��B)oB:�B��B$FB�yB!0�B�FBCIB 3B{�B&UB��B=B��B7�B�uB�_B�:BCBB"<�B?5B	�eBB�B=�B+�?B@BJbB,B%�B=�B��B��B��B
�aBPsB�B�5A��0A���B?�B	�bB�$B#l?B)@�B�DB?KB@�B��B�5B
��B@B">3B#�IB,@�B�SB@4B#��B&=(B@B:s@8��A�ܿC��A+�C��'A-��AY��B�#A>��A1JZAC�kAm]�A���AZ��A�'�A�\�A��/A��A���A�aO@�k�A���@��AI�A$;SA�DAv�)@��|@�U�A9��A�3A��}A�RX>��b?�(�A�T�A�`ZA�$�A���A��IA�W8AtY-A���@�[.@�#�@�h�Aݡ%A���A���@ƀ�A�<A��A�R�A��@A�XEA���A��PA]zRAe��@a�*@���A���A�ޝA؛NAB� A��A��,=:�\?�F#@+��@��MB	�:AMA(��A%C�A�6%C�zH@7�A���C�
�A*��C���A-�AY:B<lA@��A1�ACP�An�A�}�A[,A�{�A�xA��A�WA��IAˈ@��^A��}@��AGP�A#��A�{WAv�p@��@�V�A9sA��A�+.A��J>�f?�!=A��A�n A���A�|�A�=�A��Av7(A�7�@�� @���@�>�A�tA� zA�`�@�d�A�{A��A�
�A�}�A�Y�A���A��A[=xAg a@c{�@ъbA��^A��MA�m2ACC4A�	�A�Q�=�~?���@$�F@���B	��A�/A(�4A%��Aʂ2C�i         "   	   C                  #                        	   &                           7             
      G            /                        	                +      
      	                     "   	         :                                              !                                                         #               %                  +            5      !               #                  '                                 !            )      %      #                                                                                                         #                  #            '                     #                                                               #      %      #                  N6�9N"eNϐ|NDO���NH��O7+'O*��N�TIN��N��"NH%fO ~�OoN6�)N�|N"v�N��N��OH�~N�DN�OO�O���N���N�M�N۞N0	bO���OnN�ĄN���Nq�_N
i�O�sPO
qvN��O3��P��NNMbOSLN�/(Nj�Oi�HO'�O�
>N�{YN��9O ��O&O$>�O3�%N�KFN��O7~Nא�O 8�O0u�N��N�,�Nm�mN�4O{��N���O=7�N?�5O�xN�#�Oũ�N.O��5NK�N�UO�aN���NS�3N��X  �    		  �  
�  �  �  @  �  a  *  �  �  �  �  a  �  �  K  ~  �  m  P  �    K  �  1  �  d  
�  b  �  T  ;  a  B  L  F  �  �    j  �    y  $  !  �  4  �  �  $  -  �  |  �  �  �  s    {  �  �  f  U  e  �  �  f  �  �  q  �  �  �  ><�j<��
<t�;o�T���D�����
�D���D���D����o�ě���`B�t��o�ě��#�
��C��D�����
�T���T����o��t���C���/��t���t��ě���1��1�ě���9X��9X��w������/������P��h�o���t���P��P��P��w�,1�#�
�',1�aG��,1�,1�49X�0 Ž@��T���8Q�49X�8Q�D���m�h�L�ͽ]/�aG���O߽ixսaG��aG���o��7L��C���\)��\)���㽾v�)*-*)������������������������	

�����������������������������������
#)+#
�������<?G</'##/0<<<<<<<<<��������������=BJN[_fgigee`[NB<66=MN[bdghige[ONIIEMMMMS[gt�������tkgb[XSS���

�����������������������������dhnrt{�����������tgd"*.67CNKFC76*=BOX[b[VONB?========��������������������pt������trpppppppppp('!knz������zqqnikkkkkk����������������������

���������!#$(&#��������������������vy|�������������tpnvNOS[bhhjhfa][OHKNLNN#/9<=<:/##%-0<INQIB<:40--------��������������������kn|������������~zqikrz��������~zrnrrrrrr�����������������������������������������������������������������������������������������������������������������������������

������4BN[g���������g[0.04��������������������������������������������������������������������������������������������pt���������������{qp�
-<INOK@30
 ��z�������������zzzzzz���������������������������	�����������������������������s}������������ztroos������������������Y[htvvtttpha[VPQYYYYrt{��������yuttrrrrrSTZadmnppomidaTPOOQSQTabgjihgaTQPPMPQQQQ��������������������Y[agt�������ytng^[YY�����������������������

 ������������������������������������������qty��������������vq������������$)1669766)��������������������gt����������okgc^^gPU[alnrtrnga_UROPPPP�������������������������	

������������������������������������ ������������������������������������

������'0<IIKLII@<10$'''''' ��#'/4:730/##!�ɺɺ������Ⱥɺֺ����ֺɺɺɺɺɺɺ��t�g�tD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��нϽȽнݽ����ݽннннннннн�E\EUECE7E*E%E&E*E1ECEPEkEvE�E�E�E�EuEiE\�ݽ������������ݽݽݽݽݽݽݽݽݽݽݿ	�����۾׾Ҿ׾����	�"�&�.�1�.�"��	�I�>�0�-�,�0�=�A�I�N�V�b�o�t�y�t�o�b�V�I�A�?�7�A�M�W�Z�f�m�s�y�s�q�f�Z�M�A�A�A�A����������������������������Z�Z�W�Z�\�f�s�����������������s�f�Z�Z�y�w�m�c�i�m�o�y�������������y�y�y�y�y�y����������������������������������	�	���"��	��������ؾ���	��"�#�,�/�.�&�"�a�[�]�a�e�n�y�z�}�z�q�n�a�a�a�a�a�a�a�a���ݿٿԿݿ������������������H�?�=�H�R�U�V�^�Z�U�H�H�H�H�H�H�H�H�H�H����ĿĳııĳĿ�������������������������N�F�D�N�Z�^�g�s�u�{�s�g�c�Z�N�N�N�N�N�NàÖÓÎËÇ�ÂÇÓàçìùþÿùõìà�������������������������������������������������������������������������������߼r�f�]�_�f�r�v�����������������������r����s�f�^�^�f�s�����������ľʾҾѾʾ������������������������ĽнӽսнĽ��������H�<�?�H�Q�U�[�a�f�i�a�U�H�H�H�H�H�H�H�H�������������ĿſĿ����������������������M�J�F�F�M�Y�b�f�h�f�Y�O�M�M�M�M�M�M�M�M�S�F�8�7�;�=�D�S�l���������������z�l�_�S�(�%������(�4�A�M�W�Z�\�Y�M�J�A�4�(��޼޼���������������������"�!���"�/�;�H�H�Q�N�H�;�/�"�"�"�"�"�"��������������$�����������ù����ùϹعܹ߹ܹϹùùùùùùùùùúJ�@�:�<�D�Y�r�~�������������������~�Y�J�s�o�g�^�Z�W�Q�X�Z�g�s�v���������������s�a�V�\�a�e�n�x�z�|�~�{�z�y�n�a�a�a�a�a�a�������������������������������������������������������������������������������)�5�B�C�B�?�5�)���������<�8�<�?�I�U�_�b�n�{�}ŏŇŃ�{�y�n�b�I�<�y�w�y�����������ĿϿѿӿѿĿ����������y¦§²²²¦��x�s�v�{�����������ʼѼɼ����������������x�x�����x�v�x�����������������������ܻû��������������ûлܻ���������čćā�y�t�j�l�tāĈčėĖĕčččččč�h�g�\�[�R�S�[�h�tāĂĉĉā��t�h�h�h�h���������������������������������������������	��
�����'�4�?�?�:�4�0�'��������������������
�������
�������#���������������
���#�+�0�7�4�0�#���������������������������������������������������������	���	�����������������Z�W�N�@�A�I�N�Z�g�s���������������s�g�Z���������������������������������������������z�w�w�z�}�������������������������	���������	��"�$�.�6�;�=�;�7�.�'��	�T�S�G�E�;�.�.�.�0�;�G�T�[�`�i�f�a�`�T�T������������!�-�3�-�!� �����������4�+�4�5�@�C�M�U�V�W�M�@�4�4�4�4�4�4�4�4�������s�m�g�f�g�s������������������������ŹŭŤŠřŘŠŭŹ���������������������B�7�6�*�+�6�B�O�U�[�]�[�U�O�B�B�B�B�B�B�Y�M�D�G�M�O�Z�f�s�����������������f�Y�O�K�O�R�U�[�[�[�^�h�l�h�d�[�O�O�O�O�O�O����������#�/�H�U�^�_�U�<�/� ������ܹ����������������ù˹ϹйϹǹù����������3�.�1�:�D�D�L�r���������������r�e�L�@�3�ɺź������������úɺʺ̺ɺɺɺɺɺɺɺɼ�}���������ʼټ��������㼽��������������$�&�0�4�8�0�0�$���������S�R�G�G�C�G�S�X�`�j�`�T�S�S�S�S�S�S�S�S�нϽĽ��������������Ľнݽ������ݽн������������ĽнѽннĽ½�������������ÇÇÇÌÓàìôõìàÓÇÇÇÇÇÇÇÇE�E�E�E�E�E�E�E�E�F
F E�E�E�E�E�E�E�E�E� > � @ = F L ] 3 B e 9 Y \ g M % D  L U N O ( @ c ' 8 G S 8 3  T L  8 ` ? c ? I ] C e W = ] 7 ! L - c ) k s q "  J = G Q * - J v ] # c c m \ S , > M d    W  X  �  "  i  d  �  o  �      v  i  u  c    _  �  �  �  �  :  �  �  �  �  $  c  �  �  �  �  l    "  3  �  �    z  �  Q  �    �  �  �    P  �  a  �  �  �  �  6  V  s  �  �  �  �  �  �  �  �  �  �  K  D  �  �  @  7  �  �  �  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  =c  �  �  �  }  i  S  >  '    �  �  �  �  �  �  �  {  i  W  E      �  �  �  �  �  �  �  �  �  �  �  �  q  Q  ,    �  �  �  �      �  �  �  �  ~  7  �  �  b    �  v  $  �  �  �  �  �  �  �      �  �  �  �  �  �  l  L  ,  	  �  �  �  t  
f  
�  
�  
�  
�  
�  
�  
�  
�  
P  	�  	�  �  Z  �  �  �  �    b  �  �  �  �  �  �  �  �  �  �  v  j  c  ^  Z  U  P  J  D  >  �  �  �  �  �  �  s  Y  :    �  �  �  |  K    �  H  @  r  @  &    �  �  �  �  �  d  H  +    �  �  t  !  �  a     �  �  x  q  i  ^  Q  D  4  #    �  �  �  �  �  d  C  !   �   �  a  [  T  N  H  B  ;  5  /  (  %  #  "  !                 �  �        '  )  #    �  �  �  D  �  �  &  �    ]  �  �  �  �  �  �  �  �  �  �  �  }  h  S  >     �   �   �   �  �  �  �  �  �  �  �  �  �  f  E    �  �  �  X    �  �  2  �  �  �  �  �  �  �  �  �  �  �  �  t  ^  @    �  �  �  }  �  �  �  �  �  }  l  Y  C  *    �  �    �  �  P    �  �  �  �  �  �    .  H  Y  `  `  W  9    �  �  3  �  o  �    �  �  �  �  �  �  �  �  �  �  �  �  r  b  S  C  #  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  Y  A  #    �  �  �  6  H  I  J  J  I  G  B  >  2  &      �  �  �  �  �  �  j  O  5  O  e  u  }  }  u  h  W  <    �  �  �  A  �  n  �  Z  	  �  �  �  �  �  �  �  �  �  �  ~  s  a  P  :       �  �  �  m  c  Y  P  E  9  -  !      �  �  �  �  �  �  �  y  g  U  (  .  K  6    �  �  �  �    '      �  �  �  S  �  b  �  �  �  �  �  �  �  �  l  O  /    �  �  �  �  �  [  7  $         �  �  �  �  �  �  �  �  �  �  �  �    o  ^  C  (    d  �  �  �    *  A  J  E  ;  /  !    �  �  �  �  y  X  5  �  �  �  �  �  �  �  w  l  a  W  M  C  8  .  $         �  1  2  2  2  3  3  2  2  1  1  0  .  -  +  *  *  +  -  .  /  U  �  �  �  s  T  -    �  �  \    �  p    �    t  �  �  d  a  Z  J  6       �  �  �  f  6  �  �  �  @  �  �  Y  �  
�  
�  
�  
�  
^  
4  	�  	�  	a  	  �  2  �  K  �  c  �  l  �  ~  �  �  *  Z  ^  X  K  ?  2  &    
  �  �  �  �  M    �  &  �  �  �  �  o  \  J  6  #    �  �  �  �  �  �  3  �  �  �  T  O  K  F  A  =  8  0  &        �  �  �  �  �  �  �  �  �  �    1  :  9  +    �  �  [  !  �  �  d  �  u  �  �    F  I  \  ^  V  J  ;  %  
  �  �  �  p  @    �  �  L    �       4  @  9  5  4  5  6  6  2  (            �  T  �  L  C  8  .  =  :  ,      �  �  �  �  d  ;    �  �  �  =    $  :  F  B  8  ,    �  �  �  w  ,  �  |  +  �  `  �   �  �  �  �  �  �  �  �  �  �  �  |  Y  6    �  �  �  ~  W  1  �  v  �  �  �  w  m  _  I  -    �  �  �  �  �  S    �   �           �  �  �  �  �  �  �  �  �  �  �  �  |  m  ]  N  S  X  ^  f  m  r  s  s  m  g  `  Y  P  A  1      �  �  `  �  �  s  U  5    �  �  �  �  Q    �  �  <  �  �  8  �  8    �  �  �  �  �  �  �  y  \  E  4    	  �  �  �  �  �  �  y  j  X  J  E  E  P  \  c  b  Y  K  8     �  �  �  J  �  :  $    �  �  �  �  �  �  �  �  v  d  Q  9  "    �  �  �  �  �    !        �  �  �  �  q  J    �  �  |  =     �  u  �  �  �  �  �  �  �  �  �  �  �  z  n  _  K  0    �  �  V  �          �  �  �  �  g  A    �  �  Q    �  h  �  �  [  �  �  �  �  �  v  Y  :      �  �  �  �  �  s  V  8       �  6  �  �  o  �  �  �  ~  H      $  �  _  �  M  �     V  $      �  �  �  �  �  |  [  9    �  �  �  }  S  $  �  �  -  (  $      
  �  �  �  �  �  t  _  G  '    �  �  �  �  �  �  �  �  �  �  �  n  6  �  �  M  �      �  0  �  2  �  |  h  U  A  .      �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  d  9    �  �  t  -  �  |  !  �  q  n  z  �  �  �  �  �  �  �  �  �  x  E    �  X  �  e  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  c  M  +  	   �  s  n  i  c  ^  X  R  K  C  8  -  "      �  �  �  $  J  o      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  m  ^  N  ;  '    �  �  �  �  �  �  g  E  "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  8  	  �  �    }  �   �  �  �  �  �  �  �  �  �  v  W  4    �  �  �  �  t  a  G  -  f  e  U  :    �  �  �  �  �  �  x  b  J  4    �  �  �  2  U  [  `  f  k  o  r  u  u  p  k  e  ]  T  K  A  ?  >  <  ;  �    B  Z  d  d  Y  ?    �  �  {  A  �  z  �  �  D  8   �  �  �  �  �  �  �  �  v  Q  #  �  �  J  �  �    �  )  q   �  �  �  �  �  r  @  �  �  p  H    �  �  �  n  :    �  R  �  f  X  I  ;  -             '  /  7  <  <  <  <  <  <  =  �  �  r  Y  >  ,    �  �  �  ~  G    �  �  v  Z     �    �  �  �  �  �  �  �  �  �  |  n  ^  N  A  8  .    �  �  �  q  j  c  ]  S  H  =  %    �  �  �  �  �  n  T  9       �  �  �  �  �  �  ~  i  R  :       �  �  r  /  �  �  `     �  �  �  �  �  �  �  �  �  r  `  L  8  $    �  �  �  �  r  W  �  �  �  �  �  �  t  i  a  ]  Y  U  `  s  �  �  �  �  �  �  >    �  �  �  �  ]  5    �  �  �  j  @    �  �    �  a