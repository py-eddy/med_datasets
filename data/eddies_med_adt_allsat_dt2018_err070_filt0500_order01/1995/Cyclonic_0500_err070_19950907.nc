CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��G�z�     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�i   max       P��     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��S�   max       <�j     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?.z�G�   max       @F+��Q�     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v|          h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q@           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @��          $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �
=q   max       <�C�     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��	   max       B0�     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}�   max       B0�u     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >P��   max       C���     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@��   max       C���     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          j     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          G     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�i   max       P�2�     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?�P��{��     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��S�   max       <�j     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?.z�G�   max       @F+��Q�     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @vyp��
>     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @O            �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @��         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A-   max         A-     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���@��   max       ?�P��{��     �  _�      %                                 i   
   6      	               /         	   	            9   	               
            /      !      	      	                  
   	   P      &                                 &   .         %      	N|�OѕN���N��fO)�gN0~�NW�OOs��N�-�N�:�N���O�b�P��O�.P�#�O��dN�N��rO(��Os�`O��O�,�N=�.N6U�OF�N���O<�O7X>O1�SP5�RN�PuN�ԴNF��N �N0�O��O�"VN��zOF��O���N=fO$e�O��NڅqO�|jN���P�~N�9*OG�-O��2O��oO
OHN��O��O��N���Ot֒N���Nx}6N���O�/N�$Ob�rN��M�iO�x9O��O�r9O��N��YO��(N�ەNC?<�j<D��<#�
<t�<o;D��:�o:�o��o�o�D����o�t��#�
�49X��C���t���t���1��j��j��j��j��j��j�ě����ͼ�`B���+�C��\)�t��t��t���������w��w��w�#�
�#�
�#�
�#�
�'',1�,1�49X�49X�L�ͽT���aG��aG��u�����+��7L�������������������㽛�㽝�-���w�����������
���-��S������������������������������������//;=HHJOH;4/..//////_abmz���zmea________��������������������htxxtih[ZZ[aghhhhhhh����������������������������������������������������������������������������)-36>@COROMOPLGC=6*)@BHM[g�������sjf[NE@eL<0'$��#^n��use�����������������������#<I\n�����{R<��������������"#/<=<;/#!""""""""""����������������()16BEO[]_][OB861.*(wz������������~zupow�����������������������
#-"
����������"#,/0<?C<;/.)#""""""0<HINLIHG<:400000000�����������������[[ht������th[Z[[[[[[EHRT]amz{��zmaTLHHEE��������������������[[_ht�����th[VWYY[[n��������������recen��
"
��������������������������������������������������������������������#$/30/#����������������������������������������{���������������{{{{wz��������������}uqw��������������������������������������������������)576/.%�����ABGN[gstutog[NB;AAAAMZbn{�������yjbUQNJM��������

������Ubw�����������nbNJKU��������������������JOOSX`it�������th[OJ�
#/<HU]sneH#�����������������������|�dgt�������������trgd��������������������������)��������269BO[htvtopmh[OC<42��
#''$#
 �����45;BN[gssoeWNB@75434������ ������������~����������}~~~~~~~~" ����S[gt����������tg[VRS|�������������|x||||��������������������W[ht|tmh[WWWWWWWWWWW��������������������z���������������yvxzznaUH947<HRanz�����z)5<BNZfc[QB5)

itt�����������ztkhii!#*/36=@<:/-%#�����������������������������QUaafila^UURQQQQQQQQ�� �����������������������#�
�����������
�#�/�;�D�E�Q�U�`�U�H�/�#�����#�#�0�<�A�A�<�6�0�#������àÔÓÐÎÓàìòñìäààààààààÇ�z�n�j�n�q�zÀÇÓàáìîïðìàÓÇ�������!�"�#�!��������������������M�B�A�4�(�4�A�J�M�Z�\�Z�V�V�M�M�M�M�M�M������������������ʾ̾׾۾�߾׾ʾ�����������!�-�3�:�=�:�0�-�!�������	���	����"�.�.�9�;�;�:�.�%�"���	�.�"�	��������������	��"�)�.�5�3�.��ݿѿ������������ѿݿ������������Z�f�w�x�f�M�(�Ľ��y�f�i�����н��(�A�Z�5�,�(�!�&�(�0�5�A�N�T�W�X�Q�N�A�5�5�5�5�������v�k�^�U�A�4�&�,�A�s������������������������������������������������������H�F�@�B�H�M�U�_�\�U�H�H�H�H�H�H�H�H�H�H�z�z�x�zÆÇÓÛàáâàÓÇ�z�z�z�z�z�z���������������¿Ŀѿݿ������ݿѿĿ���������������������������������H�@�;�3�1�,���"�;�H�T�i�r�q�k�i�T�M�H�g�`�X�^�g���������������������������s�g���������)�)�*�)�����������������������������������������������˼M�E�@�;�8�@�M�T�Y�f�r�y�����t�r�f�Y�M�l�k�c�`�d�l�q�x���������z�x�l�l�l�l�l�l���	��	�
����"�/�0�4�5�2�/�"�"�������������}��������������Ŀƿ̿ϿĿ��������������"�/�;�H�Q�N�M�H�;�"��	�����t�i�Z�V�`�m�y�����Ŀ߿��޿Ŀ�������������������������������������r�p�e�Y�N�L�G�K�L�Y�e�k�r�~���~�~�w�r�r����������������������������������������������������������������������������)�$�)�)�2�6�B�C�F�B�;�6�)�)�)�)�)�)�)�)�b�b�U�L�L�U�a�b�n�u�{ńŇŌōŇ�{�n�b�bľĺļĿ���������
�'�0�:�9�#��
������ľ�r�p�h�f�r�s�~���������������~�t�r�r�r�r�O�L�B�?�@�B�C�O�P�[�h�t�x�z��{�t�h�[�O�h�c�[�Y�G�A�I�hčĜĦĮĮĲĦĝčā�n�h�'��'�-�3�@�L�V�L�D�@�3�'�'�'�'�'�'�'�'��|�x�x�z������������������������������������"�;�G�T�Z�[�T�G�;�.���������������	�����	���������-�!���$�-�F�_�l�x�����x�l�g�\�S�F�:�-��ּʼɼ����������������Ǽʼּ׼����ϻ����������ûܻ���4�@�M�X�R�C�'�����{�t�u�{ŇŏŔŠŢŭųŹżŹŭŠŔŇ�{�{�нͽĽ����������������������˽Խٽ׽ѽ������m�e�b�[�W�_�m�����������������������z�t�a�T�O�L�N�T�\�a�m�z���������������z�b�a�[�U�P�V�X�`�b�o�{ǀǄǀ�{�s�o�o�f�b�x�r�l�k�l�l�t�x�{���������������x�x�x�x�@�4�+�%�%�'�3�4�B�M�Z�r�����������r�Y�@���}�����������}������������������������E*E&EEEEE$E*E7ECEEELEGECE7E,E*E*E*E*�����������������������'�+�*�������D�D�D�D�D�D�D�D�D�D�D�EEEED�D�D�D�D���ſźŹŸŹ�����������������������������(�0�5�7�A�N�R�U�P�N�A�8�5�/�(�(�(�(�(�(��ĿĹĵĸĿ���������������������������/�*�'�/�;�<�H�U�V�a�k�a�U�P�H�<�/�/�/�/������ĿĳĦęĚĦĳĿ�����������������عù������ùǹϹҹҹϹùùùùùùùùùùϹ͹ƹιϹܹܹݹܹйϹϹϹϹϹϹϹϹϹ��n�i�n²¿��������������¿¦��%�/�6�0�"��	�����ھϾʾ׾����	�Ó��{ÀÇÐÙàìù��������������ìàÓàÔÓÍÇÀ�{ÇÓàìðøõùýùìààE�E�E�E�E�E�E�FF$F1F7F1F(F$FFFE�E�E�G�B�>�D�S�`�l�y���������������y�l�`�S�G��������ݽٽݽ��������������ù��ùùϹܹ���ܹܹϹùùùùùùù� 5 ; I / / F M < 1 | n  H ' ^ X 7 9 3  F ; > S & f E B 7 F / > Y L ^ 5 f r + , d : & F Q p @ H ? R ( c 5 : ] , : U + E 8 J \ ` d V R ( L � I l j    �  �  �  �  p  W  y  �  �  �  �  �  �  '  �  �  ;  �  l  �  �    _    :    A  �  �  [  �  �  ~  H  q    �    �  �  Z  u  >    2  �  �    �  �  ?  o  �    ~  �    �  }  �  N  �  �  K  %  �  |  �  O  �    �  w<�C�����<o;��
�ě���o�D�����
�ě��o��`B������xռ��㽅��o���������+�<j��㽏\)��`B��h���+���0 Ž�㽸Q�,1�#�
�#�
�,1�0 ŽD���y�#�49X��%��{�@���t���o�D������L�ͽ�%�@���+��hs�y�#�q���y�#�
=q��+�ȴ9����
��\)��-��񪽼j�ě����罧�
=���   �������ͽ����vɽ�FB�{B2�A��	A���Bu�B��BJ�B�0B,$�B^eB0�B	~�B%��B��B&v�B�B:�B�BNWB �CB=BV�BX�B�VB#�B@�A��B�B׊B*��B$;�B"��B�oB��B�jB��BE�B�SB ��B�=B�vB�aB��B�B(R>B#��B(wBB�>BZBY�B�B
��B!o�B�OB��B1BdoBئB
ґB'B
�B
��B��B��B@�B`B<NB�B
w�B%B�&B^�B�B��B�	A�}�A��qB�3B��B?�B��B,XB��B0�uB	B�B%��B�tB& �B?mB;4BASBA�B ��B�B@�B@B<B"�B@%A��B@�B,B*@B$A7B"�,BB[B��B��B��BK�BLB �B�B�.B��B�B�mB(�B#ūB(�BB|BA	Bn�BA�B
ʤB!GB��B��B<B7�B�+B
��BD}B	�ZB
�5B>�B��B@-B-rB�
B�8B
�yB��B�CBE�B��A��A��A��A˄�A�2�@aErA;�oAN�H@lr'A_5�A\:A|OA,R�A�)�A��SA��FA��A�/Az8�B3A�\�A���A��A���@��3@���A�$As�xA���Aq�/@�o�?�waA��AҦ�A�O�A�~�A��g@�)A��*A܍�?�3AG��AavzAYO�@�Lz@�ex@���A��VA$%RA��A��uB��@��@��A�U�C��A��C�8�A��A�M�A��A��A�z&>P��>�T�A�!�AY�;A�Q�A�gcC���A;,A/?�>�V�A���A���A�%xAˈ�A�L�@c�A:�xAN��@o�rA_(�A[�Ay
_A&��A��kA�zA�hA�kLA�~/Ay]BE2A���A�xxA��>A��1@܈�@�$.A�u�Asy�A��qAsi@�n�?�A���AҖ�A׆A��NA�z?�c�A�MAۼ�?�T1AGI�Ac&AY�@tb�@��c@˫�A���A#AA��5A��B�[@�5R@��A��C��A��C�@A���A�+A�nA�bDAካ>@��>�~[A�AX�A�I�A�kiC���Av�A0�>��W      &                                 j   
   7      	   	            0         	   
            :   	                           /   	   !      	      
                  
   
   Q      '                                 &   /         %      	      #                              #   B      C                     #                        .                              !                     )         %            %                                           !                                                      9      G                                             .                                                   #         %                                                       !               N|�O�B\N���N��fN�E�N0~�NW�OND�N�-�N�:�N���O�|�PyqN���P�2�O��dN�N��rO�OV�wO�q�OV��N=�.N6U�OF�N���N��O��O1�SP5�RN�PuN�ԴNF��N �N0�O��O�"VN��zN��O9O`M��N�n�O���NڅqO]��N���O��pN�9*O;~EO��2O��oO
OHN��O��&Nު�N�C�Ot֒N���Nx}6N���O��zN�?OP��N��M�iO�x9O��O��O��N���Oi\�N�ەNC?  �  :  O  �  �  i  �  �    �  �  ,    R  u    6  �  �  ,    !  �  �  E  \    �  %  �  ?  c  9    �  <  W  �  �  �  w  �  u  L  4  	  t  \  ,  �  ,  2  Z  
~  �  �  �  �    �  �  �  b    {    �  H  �  �  �  X  )<�j;D��<#�
<t�;�o;D��:�o�D����o�o�D����`B��h�D���D����C���t���t���9X���ͼ��ͽ'�j��j��j�ě����������+�C��\)�t��t��t��������<j�aG��#�
�P�`�,1�#�
�,1�'D���,1�0 Ž49X�49X�L�ͽT����7L�e`B�}󶽅���+��7L�������-���-���㽛�㽛�㽝�-���w��������1��-��S������������������������������������//;=HHJOH;4/..//////_abmz���zmea________��������������������htxxtih[ZZ[aghhhhhhh��������������������������������������������������������������������������������)-36>@COROMOPLGC=6*)Q[gt��������tg[NGGKQ
#0Ptwxlf[I<0, 
�����������������������#<]n�����{Q<0��������������"#/<=<;/#!""""""""""����������������46BDOR[\^\[OB;630,*4rz��������������zvrr�����������������������

���������"#,/0<?C<;/.)#""""""0<HINLIHG<:400000000�����������������[[ht������th[Z[[[[[[HHTamxz�zxmaTQKHHHHH��������������������[[_ht�����th[VWYY[[n��������������recen��
"
��������������������������������������������������������������������#$/30/#����������������������������������������{���������������{{{{��������������}{���������	
	�����������������������������������������������)10,$ ����ABGN[gstutog[NB;AAAAO\bn{��������{nbUPLO��������

������Ubnt{�������{nfTQQRU��������������������Xahjt�������xth[QQTX�
#/<HU]sneH#�����������������������|�dgt�������������trgd�����������������������������������66BO[hmolhd[ONDB=666
#&&##
�45;BN[gssoeWNB@75434������ ������������~����������}~~~~~~~~" ����T]gt���������tga[WST�����������}z��������������������W[ht|tmh[WWWWWWWWWWW��������������������z���������������yvxzznaUH947<HRanz�����z)5:BU[^VOB5)%itt�����������ztkhii!#(//0248:/.'# !!���	���������������������������QUaafila^UURQQQQQQQQ�� �����������������������/�#�������������
��#�/�3�<�>�F�H�3�/�����#�#�0�<�A�A�<�6�0�#������àÔÓÐÎÓàìòñìäààààààààÓÇÇ�z�q�y�zÇÎÓÕàééààÓÓÓÓ�������!�"�#�!��������������������M�B�A�4�(�4�A�J�M�Z�\�Z�V�V�M�M�M�M�M�M�����������������ʾ˾Ծʾ���������������������!�-�3�:�=�:�0�-�!�������	���	����"�.�.�9�;�;�:�.�%�"���	�.�"�	��������������	��"�)�.�5�3�.�������������Ŀѿݿ������������ݿѿ����w�v�z���нݽ���A�M�T�^�M�A��ݽ������5�0�(�#�(�)�5�6�A�N�O�P�R�N�A�>�5�5�5�5�������w�k�_�A�4�'�.�A�s��������������������������������������������������������H�F�@�B�H�M�U�_�\�U�H�H�H�H�H�H�H�H�H�H�z�z�x�zÆÇÓÛàáâàÓÇ�z�z�z�z�z�z�������������ĿĿѿݿ߿����ݿѿĿ�����������������������������������;�6�4�(�#�!�/�;�H�T�a�m�o�m�g�e�Y�T�H�;�u�s�m�u�������������������������������u���������)�)�*�)�����������������������������������������������˼M�E�@�;�8�@�M�T�Y�f�r�y�����t�r�f�Y�M�l�k�c�`�d�l�q�x���������z�x�l�l�l�l�l�l���
�����"�+�/�2�4�0�/�"����������������������������������ÿĿɿƿĿ������������"�/�;�H�Q�N�M�H�;�"��	�����t�i�Z�V�`�m�y�����Ŀ߿��޿Ŀ�������������������������������������r�p�e�Y�N�L�G�K�L�Y�e�k�r�~���~�~�w�r�r����������������������������������������������������������������������������)�$�)�)�2�6�B�C�F�B�;�6�)�)�)�)�)�)�)�)�b�b�U�L�L�U�a�b�n�u�{ńŇŌōŇ�{�n�b�bľĺļĿ���������
�'�0�:�9�#��
������ľ�r�p�h�f�r�s�~���������������~�t�r�r�r�r�O�M�F�I�O�[�]�h�n�t�u�{�v�t�h�[�O�O�O�O�[�Z�P�M�O�[�h�tāčėĚğęčĂā�t�h�[�'�&�'�0�3�@�L�S�L�B�@�3�'�'�'�'�'�'�'�'���}�~��������������������������.�"�������"�.�G�T�X�Z�V�T�L�G�;�.��������������	�����	���������-�!���&�-�F�S�_�l�x���x�u�k�c�W�F�:�-��ּʼɼ����������������Ǽʼּ׼�����Իλл߻����4�M�N�M�I�@�'������{�t�u�{ŇŏŔŠŢŭųŹżŹŭŠŔŇ�{�{���������������������ĽɽнӽؽֽнĽ��������m�e�b�[�W�_�m�����������������������z�t�a�T�O�L�N�T�\�a�m�z���������������z�b�a�[�U�P�V�X�`�b�o�{ǀǄǀ�{�s�o�o�f�b�x�r�l�k�l�l�t�x�{���������������x�x�x�x�Y�M�@�4�1�*�)�*�/�4�@�M�j�}��������r�Y����������������������������������������EEEEE&E*E7ECEDELEFECE7E*EEEEEE�����������������������'�+�*�������D�D�D�D�D�D�D�D�D�D�D�EEEED�D�D�D�D���ſźŹŸŹ�����������������������������(�0�5�7�A�N�R�U�P�N�A�8�5�/�(�(�(�(�(�(��ĿĺķĺĿ���������������������������/�-�)�/�<�C�H�O�U�a�U�N�H�<�/�/�/�/�/�/������ĿĳĦĚĚĦĳĿ�����������������عù������ùǹϹҹҹϹùùùùùùùùùùϹ͹ƹιϹܹܹݹܹйϹϹϹϹϹϹϹϹϹ��n�i�n²¿��������������¿¦��%�/�6�0�"��	�����ھϾʾ׾����	�ÓÁ�}ÂÇÓàìù����������������ìàÓàÔÓÍÇÀ�{ÇÓàìðøõùýùìààE�E�E�E�E�E�E�FFF$F'F$FFFE�E�E�E�EٽG�C�A�@�E�S�`�l�y�������������y�l�`�S�G��������ݽٽݽ��������������ù��ùùϹܹ���ܹܹϹùùùùùùù� 5 : I / # F M [ 1 | n  J / Z X 7 9 4  8 < > S & f D A 7 F / > Y L ^ 5 f r  5 i , % F M p : H > R ( c 5 + R + : U + E 4 @ Z ` d V R ' L x ; l j    �  .  �  �  �  W  y  U  �  �  �  x  �  �  �  �  ;  �  B  �  -  �  _    :      \  �  [  �  �  ~  H  q    �      �  F  �  �    �  �  �    �  �  ?  o  �  �  )  �    �  }  �    �  �  K  %  �  |  �  O    �  �  w  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  A-  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    (  2  9  :  1  !    �  �  �  3  �  a  B  �  �  h    O  D  9  .  "        �  �  �  �  �  �  �  |  i  U  B  /  �  �  z  q  h  ^  R  G  <  0  &      
    �  �  �  �  s  �  �  �  �  �  �  �  �  �  �  �  �  o  T  7    �  �  r  �  i  c  \  V  P  E  +    �  �  �  �  �  p  U  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  r  i  _  U  K  B  8  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    G          	    �  �  �  �  �  �  �  �  �  �  �  �  y  k  �  �  �  �  �  �  �  �  �  �  �  �  {  f  S  ?  +      �  �  �  �  �  �  �  z  o  e  Z  R  N  I  E  @  E  M  U  ^  f  �    +  *  #    �  �  �  �  �  �  �  �  �  b    �  6   �  p  �  �        �  �  h    �  u  �  �  t  <    d  {  v  B  I  P  Q  R  O  M  O  U  T  M  C  5  $    �  �  �  �  j  m  l  G    �  �  4  �  �    �  `  _  7  �  �  \  )  �   �        �  �  �  �  �  �  j  Z  ?    �  �  �  �  ^  .   �  6  <  B  @  8  1  ,  &                 �  �  �  �  �  �  �  �  �  t  b  M  8  !  
  �  �  �  �  i  <     �   �   y  �  �  �  �  �  �  �  w  k  ^  O  ?  *    �  �  �    ;   �  '  +  +  %      �  �  �  e  '  �  �  f  #  �  �  g  �  $            �  �  �  �  �  x  X  4    �  �  �  �  �  �  �  �  �  �  �             �  �  �  i    �  3  �  e  W  �  �  �  �  q  _  M  :  '       �  �  �  �  �  �  k  U  >  �  �  �  �  |  m  ^  I  2    �  �  �  �  �  _  >       �  E  D  B  >  8  1  &    	  �  �  �  �  �  �  ~  v  r  v  y  \  ;    �  �  �  �  �  �  �  v  c  M  ;  /  #        .                
    �  �  �  �  �  �  �  �  a  4    �  �  �  �  �  �  �  �  q  [  @    �  �  �  R  
  �  e   �  %             �  �  �  �  �  �  �  �  �  �  �  k  F     �  �  �  �  �  g  Q  ;    �  �  }  1  �  b  �  I  �     �  ?  8  0  '        �  �  �  �  �  �  �  �  r  \  H  :  ,  c  \  T  L  E  C  A  >  =  <  <  ;  3  %    
  �  �  �  �  9    �  �  �  �  �  �  �    q  a  Q  @  0       �   �   �          �  �  �  �  �  �  s  Q  /  	  �  �  �  ^  8    �  �  �  �  �    e  L  3    �  �  �  �  [  3    �  �  �  <  /  !  	  �  �  �    X  0    �  �  �  �  �  8  �  �  �  W  U  Q  I  =  1  #    �  �  �  �  P    �  +  �     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  a  B  !  �  �  �  d  0  �  �    j  �  �  �  �  �  �  �  �  �  o  (  �  c  �    9  0  �  h  n  t  y  |  |  s  j  R  8  �  �  ,  �  �  �  L     �   �  *  6  M  b  w  �  �  �  w  \  /  �  �  k  !  �  |  
  �  C  f  r  r  i  Z  E  /    �  �  �  w  ;  �  �  X  �    �  1  L  8  $    �  �  �  �  �  �  {  Y  7    �  �  �  x  V  3    4  3  0  -  *  $    �  �  �  �  �  �  S    �  �  �  b  	  �  �  �  �  �  �  �  �  q  ]  K  ;  -  #  "  =  \  �  �  a  `  c  i  h  k  t  r  c  L  ,    �  �  [  3    �  �  �  \  R  I  ?  5  %      �  �  �  �  �  �  �  �  w  h  X  I    +  $    	  �  �  �  �  �  m  C    �  �  U  �  �  )  �  �  �  �  v  \  :    �  �  �  �  �  }  f  R  <    �  a  �  ,      �  �  �  �  �    T    �  �  c  0    �  �  Y  a  2      �  �  �  �  �  �  �  �  }  Y  0     �  �  f  9    Z  T  M  =  +    	  �  �  �  �  �  �  y  c  N  <  *    �  	�  
S  
v  
~  
r  
R  
$  	�  	�  	�  	D  �  �  �  z  �  (  <  3  �  �  �  �  �  �  �  z  g  R  :    �  �  �  �  �  }  g  n    �  �  �  �  ~  S    
�  
�  
^  
  	�  	r  	  �  B  �  �  �  �  �  �  �  w  `  G  .    �  �  �  �  �  �  W  )  �  �  �  .  �  �  �  �  i  �  �  �  m  O  /    �  �  �  �  i  ,  �  �      	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  k  �  �  �  �  �  {  h  S  =  $    �  �  �  q  1  �  �  h  $  r  �  �  �  �  �  e  :  
  �  �  R    �  {  +  �  t    �  �  �  �  �  �  �  �  v  M    �  �  S    �  Y  �  �  >  �  ]  a  Y  I  5  !    �  �  �  u  6  �  �  p  9  L    q  4    �  �  �  �  �  �  �  �  �  �  p  6  �  �  �  f  5    �  {  t  m  g  O  4    �  �  �  �  �  h  H  &    �  �  �  w    �  �  �  ~  `  E  ,    �  �  �  �  T    �  )  �    D  �  �  �  �  �  �  e  A    �  �  �    @  �  �  ,  �  �  �  =  G  >  &    �  �  �  K    �  �  ;  �  �    r  �  ?  8  �  �  |  m  ]  I  /      �    '  "  ,  :  F  /  �  	  A  Y  I  �  o  '  �  t    �  `  L    �    �  �  d  �  &  o  �  �  �  �  �  �  v  a  H  %    �  �  }  ;  �  �    (    X  I  ;  ,        �  �  �  �  �  �  �  �  �  �  �  �    )      �  �  �  �  �  s  Z  B  ,      �  �  �  �  �  �