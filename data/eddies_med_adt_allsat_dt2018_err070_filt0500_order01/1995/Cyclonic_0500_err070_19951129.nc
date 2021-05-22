CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�����+       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�Tg       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�w       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�        max       @F7
=p��     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @v|(�\     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @O            �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @��           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �!��   max       =+       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��9   max       B0       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B/³       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >ʧz   max       C���       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��9   max       C��!       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          O       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�Tg       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ȴ9Xc   max       ?��+j��g       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =�w       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @F.z�G�     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v|(�\     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @O            �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @��           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?y�_o�    max       ?��+j��g     �  ]            >   /                                             7      
                  $   N         $                  4         $         *   *   1      5         
      /          (                           1      %         "N/�OTgN�NP��PCK�N�nP��N�]�N-h�N/0N^:1N��O,\nOX�(N tO.u{O���N*�N4�P�TgN�f\O1�vN���OvC�OEߧN-{Oŀ�O�f>P���N�F�NUP	��O�eOp�O�/O�:3O5��O���NFNa�,O��SO��"N�f�O�*JOi�OڅOP��O�M���O0҈OElO�EO��;O~BKO�
�N�>�N\�N��RO5*N�<YN�"tN���Nk�PeO ��OPV�NVc1N��#N���=�w<��<u<T��<D��<49X<t�;D��:�o��o��o�o��o�o�49X�T���T���T���T���e`B��C����
���
���
��9X��9X�ě����ͼ��ͼ��ͼ��ͼ�����`B��h��h��h�����o�t���P��w��w�#�
�0 Ž0 Ž8Q�8Q�8Q�8Q�<j�<j�<j�@��H�9�H�9�L�ͽP�`�]/�ixսq����%��\)��hs��hs���-���
���T��E������������������������������		��������LNS[gstx����tmg[PNLL����������������������������,-'���������
	������������FNIMPUns�����naUJLF!#/6<<<<7/#!!!!!!!ehltv~���xtnmheeeeee������������������������

�����������U[ggkg][XQUUUUUUUUUU����������������������),&$��������������������������������
�������
#/=JLHA</# 
�
��������������������<BOSWQQONMB=<<<<<<<<���#<{������b<0�����������������������_fmz����������zpmia_�����

�����������*6CGQW[[O6*

`demz�������zrmba]]`chot||vth`ccccccccccMYit���������tdWOKJM��������������������koz�������������xpik���������������������������������������������������������������

����RTamvz���zpma_TSPNR���������������������������	 �������/8=>BJO[mqg^[WOB;61/��������������}{��������� ��������������������������������������
���������������������������{��������������������������������������{y�{�����������������~{AO[htu}����~thbOD>=A,/;<HJPTUXVUH<4/+'%,LUn�����������maULEL�������#!������������������������������~leaadnz���������45BNTX[\_^VNB?540044glt������������y[\agF[ht}��tmh[WVTOKGEFU\bhja`UH</"#/<HU)5?>E>CA;)�������������������������������������������������������R[]gt���������tgb[SR������yy��������������������������������z{~������������{yzzzBBDLNPUWPNLBBBBBBBBBMSV\cit����������tNM���	���������<ABB?<7/##-/<#049:80.#"rt���������tkmrrrrrr:<@HKUaaljaaUUHA<9::E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¦²³·¸¸²­¦�����������������������������������������/�����<�L�U�n�o�y�ÂÄÃ��p�U�H�/���������s�Z�A�5�7�A�Z�s�������������������������������	��������������������������g�Z�5�$��5�N�g���������������������<�8�1�<�E�H�U�V�a�i�g�a�U�H�<�<�<�<�<�<�H�B�<�5�<�H�O�U�Y�a�c�a�U�K�H�H�H�H�H�Hìäà×ÞàìîùôììììììììììEEEEE*E0E7EBECEGECE7E3E*EEEEEE�f�]�f�f�s�x�����s�f�f�f�f�f�f�f�f�f�fÓÑÍÇÈÏÓàçìðù������ýùìàÓ���s�l�_�c�g�s�����������������������������������ʼּۼּμʼ��������������������"��"�/�2�/�;�C�T�a�m�u�l�a�[�T�H�;�/�"�1�)�%�#�$�)�6�B�O�[�f�k�g�h�q�g�[�O�B�1�H�H�H�K�P�U�a�d�a�\�]�U�H�H�H�H�H�H�H�H�a�Y�^�a�n�zÇÑÇ�z�t�n�a�a�a�a�a�a�a�a���������Z�F�/�.�@�Z�s������������������Y�X�P�M�G�H�M�Y�f�k�o�r�s�r�f�_�Y�Y�Y�Y�O�6�*��'�)�5�=�B�O�T�T�[�[�V�[�\�[�U�O�Y�P�V�Y�Y�`�f�r��|�~�r�o�f�Y�Y�Y�Y�Y�Y�	�����������	���)�/�0�0�.�"��	�����޿��������(�/�5�8�5�4�(�������������������ý������������������������׾ʾ˾ʾ̾׾�	�"�%�-�1�.�"��	������������������������$�0�6�2�+�!������������������'�Y���ɺ��ٺ����r�L��ֺҺҺҺֺܺ������������׺ֺֺֺ־�����~��������������������������������B�A�0�0�,�,�N�[�tĎĚĬĸĮĦčā�[�O�B�ּϼʼ����������������������üּ̼ܼ�����
�����"�%�/�2�;�>�H�H�A�;�/�"��_�S�G�S�_�l�x�������ûɻ»����������l�_����лû����ûлܻ���'�/�1�/�'����������x�s�x�}�������������ûȻƻû������S�S�F�F�O�l�x���������������������x�i�S�Ŀÿ¿Ŀɿѿӿݿ߿ݿԿѿĿĿĿĿĿĿĿĿm�f�l�m�y�������������y�m�m�m�m�m�m�m�mŇ�{�v�yŔŠŭŹ����������������ŹŭŠŇ�U�Q�@�<�2�0�4�<�U�b�n�{�ŀ�|�{�v�n�b�U�������������
��#�(�.�#�!����
�������d�T�5�+�*�2�;�T�m�z�����������������z�d�Ŀ����������Ŀѿݿ�������������ݿѿĹù������������Ϲܹ�����������ܹ�E�E�E�E�E�E�FFFF$F1F6F7F8F1F-F$FFE��T�G�5�4�6�H�T�a�m�������������������z�T�ɺ������������ɺ����!�&�(�!����ֺɺe�`�e�q�r�~�������~�~�r�e�e�e�e�e�e�e�e������׾ʾ������������������þʾ׾�����������������*�C�F�D�C�8�6�*����ŭŠœŊŊőŠŭŹ����������	�������ŭ������������������� � ������������������������ĿпѿԿԿӿѿĿ���������������åÔ�z�u�zÇÓàù������������������ùå��ۼ�����������������������������������������������������������������������������$�$�$�$� ���������������������	���������	��ììùûþûùìéàÝ×àììììììì���
���������
��!�#�+�/�4�/�#�����л̻û����������ûлܻ��������ܻл��/�,�#����#�/�<�>�<�0�/�/�/�/�/�/�/�/������¦�u�p�q�~¦¿����������	��:�4�:�G�S�`�l�y�����������y�l�g�`�S�G�:D{D�D�D�D�D�D�D�D�D�D�D�D{DoDbDZDaDoDqD{�������������ĽȽĽ�������������������������������������������������������������E�E�E�E�E�E|E�E�E�E�E�E�E�E�E�E�E�E�E�E� j E R   R 8 K 0 @ Z Z 8 = l \ _ ) | t a M T R D < ` ) 9 U 5 > 5 O = [ ; G E > > - $ p 0 " 6 0 : K q X E ' _ / N o f _ J R U 0 c P l J @ V 4  r  e  6  D  �  �  �  �  L  y  �  &  y  =  .  �  �  �  t  �  �  �  �    �  9  �       �  >  w  @  M  S  �  �    3  h  F  �  �  A  �  �  +  �  9  J  �  �          �    �  �  �  �    R  �  Z  �  q  �  =+<D��<t��H�9�C�;D����t���`B�D���t��u���
���
���㼋C���`B�,1���
���
��C��o��h�����0 Žt����ͽ<j�y�#���o��h�y�#�<j�#�
�P�`�H�9�,1����t��,1�����o�,1���1��j�u�ȴ9��t��L�ͽe`B�ixս�j���w���
��9X�e`B�ixսy�#���w������-���������������h��j���;!��B�BE B	R�B5cB��B��B�.B4B9`B�B*fB�~B�B��B��B)�B�B3zBt�B&�B �cB �B#hIB0A�CB�B��BoB��Bq�B�>B�B#�zA��9By�B��B��B!�B��B*ҾB�"Be�B�B�B��B�B�BlvB#��B"hBH�B�B
��B��B��BO5B.�B�B��B	�B
�B��B)�%BF<B	�BQoB�6B%��B
U�B��BHB@B��B�NB6�B�"B?QB7�B?#B��B?GBǂB�]BEB�.B>HB��BA�BH�B&��B �6A�+#B#hB/³A��>B�B��B�B�,B��B�IBiB$?�A���BEFB��BD�B �B�B*��B�B�JB9�BB5�BG�B�HB��B#@B"B�B?%B˂B
��B�0B�9B�bB.C�B6zB:UB	�0BB�B)�+B?�B
:�BB*B��B%�hB
@MB��C�1*A�
�A�XA��?A��A�.!A�3A�A��A�HYC���AB�;A���A���@���A�ZA��A�e�A��&A��c@���A؝@ݩSA[n�A�uCA#jlAX��BW�?�Z�@GGSAG�'A��@�G�A��t@�D@���@�@�@�n�Az�RAn�A��;A��cA�PoA��%A{y<>ʧzC���A� �@EG�?��AQ��A�K5A��mA���AvI}A�H�A�wA�a�B�(AZMḀyA��@���A�ӠA� ZAs\C��-A#hA�#}C�RC�:QA��A��3AāPA�M�A��$A�~�A�v�AĈ�A�~�C��$AB�QȦ�A���@�$ZA�Q�A�s�AŅrA��A�^�@�
�A��@ݿ�AZ�tA��:A"��AYq�B��?ٳ�@E�AF��Aۜ1@���A��D@�tK@��@��S@��A{�Am�8A���A�w�A斆A���A{s0>��9C��!A��	@4QT?��AN�bA�q�A���A�f�AuORA��A��A��B�xAZxÂ�A��@�g�A�~CA�}�A+WC��_A#�A��C��            ?   /   	            	                              7      
                  %   O         $                  5         %         +   *   2      6               /           (                           2      %         #            !   1      -                                       C                     !   #   ;         '            !      %                  #      %      )   )            #   #      #                           '                                    -                                       C                        #   ;         '                                                   )               #      #                           #               N/�Nۚ7N�NO��VO���N;iP��NZ؏N-h�N/0N+W�N��O?O:`hN tN���OF�6N*�N4�P�TgN�f\O1�vN~�ODa�O!�N-{N�cO�*P���N�F�NUP	��O�eN��O�oO��O5��Oa|NFNa�,O��KO��"N�f�O��LO2�O@�.OO�^&O�M���O0҈OElOGO��ZO~BKO�
�N�>�N\�N��RO�N�<YNL�N���Nk�O�O ��OPV�N8^N��#N���  �  �  `  �  �  c  �  �  �  A  4  �  +  �  2  �  �  3  m  w  �  S  �  �  �    �  R  e  W  �  �  �  �  ;  r  �  �  �  @  �  �  �  }  �     �  f  �  (  �  �  �  M  �  &  �  [  1  �  �  �  �  �  �  �  	�  @  �  /=�w<�/<u�o�t�<o<t�:�o:�o��o�D���o�ě��t��49X�u��t��T���T���e`B��C����
��1��j�ě���9X��P�������ͼ��ͼ��ͼ�����`B�o��P�+���D���o�t��#�
��w��w�<j�L�ͽ}�8Q�u�8Q�8Q�<j�<j����L�ͽH�9�H�9�L�ͽP�`�]/�u�q�������\)��hs���㽝�-���
���E����������������������������������������LNS[gstx����tmg[PNLL�������������������������
�����������������������FNIMPUns�����naUJLF###/9:4/#"########ehltv~���xtnmheeeeee������������������������

 �����������U[ggkg][XQUUUUUUUUUU�����������������������)*$"�����������������������������	������
#/7<FIE></#
��������������������<BOSWQQONMB=<<<<<<<<���#<{������b<0�����������������������_fmz����������zpmia_�����

�����������"*6CLOSTOC6*`abgkmz�����~zmia_^`chot||vth`ccccccccccghst���������tohd`gg��������������������koz�������������xpik���������������������������������������������������������������

����PTXampz�zwmkaXTSPPPP������������������������������������/8=>BJO[mqg^[WOB;61/������������������������� ����������������������������������������
��������������������������{��������������������������������������}|���������������������JO[hotz~{zxth[SOKIHJ,/;<HJPTUXVUH<4/+'%,S]anz��������naUPNOS�������#!������������������������������~leaadnz���������45BNTX[\_^VNB?540044ptx������������tsoopI[ht{���|th[XXVOLIGIU\bhja`UH</"#/<HU)5?>E>CA;)�������������������������������������������������������U[agt���������tgg[VU������yy��������������������������������z{~������������{yzzzBBDLNPUWPNLBBBBBBBBB[_agt����������t[RW[���	���������<ABB?<7/##-/< #038820/#"        rt���������tkmrrrrrr:<@HKUaaljaaUUHA<9::E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¦ª²´¶´²§¦�����������������������������������������/�%��!�'�/�<�H�U�a�n�y�z�y�s�n�a�U�H�/�s�g�Z�M�I�K�N�Z�g�s�}�����������������s������������������������������������������g�Z�5�$��5�N�g���������������������<�<�<�<�H�U�a�g�b�a�U�H�<�<�<�<�<�<�<�<�H�B�<�5�<�H�O�U�Y�a�c�a�U�K�H�H�H�H�H�Hìäà×ÞàìîùôììììììììììEEEEE*E3E7E=ECEDECE7E0E*EEEEEE�f�]�f�f�s�x�����s�f�f�f�f�f�f�f�f�f�fàÓÏÌÌÓÓàåìîù��������üùìà���z�s�n�h�a�e�g�s�������������������������������ʼּۼּμʼ��������������������H�@�;�5�3�;�E�H�T�a�m�s�m�j�a�]�W�T�H�H�O�D�B�6�)�%�&�)�,�6�B�O�[�a�g�a�f�_�[�O�H�H�H�K�P�U�a�d�a�\�]�U�H�H�H�H�H�H�H�H�a�Y�^�a�n�zÇÑÇ�z�t�n�a�a�a�a�a�a�a�a���������Z�F�/�.�@�Z�s������������������Y�X�P�M�G�H�M�Y�f�k�o�r�s�r�f�_�Y�Y�Y�Y�O�6�*��'�)�5�=�B�O�T�T�[�[�V�[�\�[�U�O�Y�T�Y�Y�Z�a�f�p�r�x�{�r�l�f�Y�Y�Y�Y�Y�Y�	��������������	���%�+�+�)�"��	����������������(�,�6�5�/�(�������������������ý��������������������������������	�������	���������������������������$�0�5�2�*� �����������������'�Y���ɺ��ٺ����r�L��ֺҺҺҺֺܺ������������׺ֺֺֺ־�����~��������������������������������B�A�0�0�,�,�N�[�tĎĚĬĸĮĦčā�[�O�B�ּϼʼ����������������������üּ̼ܼ���"�������"�*�/�9�;�C�;�5�/�"�"�"�"�x�q�l�_�Y�W�_�l�q�x�������������������x���ܻû������л������'�,�+�'������������x�s�x�}�������������ûȻƻû������x�u�e�_�\�Y�X�\�_�l�x�����������������x�Ŀÿ¿Ŀɿѿӿݿ߿ݿԿѿĿĿĿĿĿĿĿĿm�f�l�m�y�������������y�m�m�m�m�m�m�m�mŔŇ�x�|ŇŋŔŠŭŹ������������ŹŭŠŔ�U�Q�@�<�2�0�4�<�U�b�n�{�ŀ�|�{�v�n�b�U�������������
��#�(�.�#�!����
�������l�T�9�/�/�;�T�m�z�������������������z�l�Ŀ����������Ŀѿݿ����������ݿҿѿĹù����������ùϹܹ��� � �������ܹϹ�E�E�E�E�E�E�FFFF$F1F6F7F8F1F-F$FFE��T�H�D�;�:�:�=�H�T�a�m�z���������z�m�a�T�ɺ������������ɺ����!�&�(�!����ֺɺe�`�e�q�r�~�������~�~�r�e�e�e�e�e�e�e�e������׾ʾ������������������þʾ׾�����������������*�C�F�D�C�8�6�*����ŭťŠŚśŠţŭŹ������������������Źŭ����������������������	�����������������������ĿпѿԿԿӿѿĿ���������������åÔ�z�u�zÇÓàù������������������ùå��ۼ�����������������������������������������������������������������������������$�$�$�$� �����������������������	�	�������	��ììùûþûùìéàÝ×àììììììì��������
���#�(�#��
�����������������л̻û����������ûлܻ��������ܻл��/�,�#����#�/�<�>�<�0�/�/�/�/�/�/�/�/¿¦�y�t�w¦¿����������������¿�:�4�:�G�S�`�l�y�����������y�l�g�`�S�G�:D{D�D�D�D�D�D�D�D�D�D�D�D{DoDbDZDaDoDqD{�������������ĽŽĽ�������������������������������������������������������������E�E�E�E�E�E|E�E�E�E�E�E�E�E�E�E�E�E�E�E� j = R  * ? K 2 @ Z X 8 A d \ H ( | t a M T U C A `  7 U 5 > 5 O % A D G ? > > + $ p 8  # 0 > K q X E  ^ / N o f _ G R : 0 c M l J ? V 4  r  
  6  i  4  >  �  x  L  y  e  &  P  �  .  !  �  �  t  �  �  �  �  �  m  9  �  �    �  >  w  @  �  -  >  �  �  3  h  �  �  �  �  u  �  +  �  9  J  �  �  �  �      �    �  l  �  h    R  6  Z  �  \  �    A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  �  �  �  �  �  �  �  �  ~  s  g  Z  M  A  6  +    �  �  �  r  u  |  {  �  t  f  S  5    �  �  V    �  x  6  1    '  `  R  D  5  -  &               �  �  �  �  �  �  �  �  �  3  \  {  �  �  �  w  b  K  .    �  g  �  j  �  �  �  G  �    G  e  �  �  �  �  �  �  �  �  \  %  �  �  <  �  >   �  Y  T  O  N  N  Q  Y  b  Q  =  (    �  �  �  �  �  |  _  B  �  �  �  �  �  �  �  �  �  �  �  �  �  w  e  W  <    H   ?  �  �  �  �  �  �  �  �  {  n  `  O  <  (    �  $  `  \  P  �  �  �  �  �  �  q  b  R  C  4  &    
   �   �   �   �   �   �  A  H  O  T  Y  Y  I  9  )      �  �  �  �  �  �  ~  g  P    (  0  4  6  4  -  !    �  �  �  �  ^  4    �  �  v  >  �  �  �  �  �  �  �  �  �  �  �  x  o  c  Q  ?  -       �    #  )  *       �  �  �  �  |  c  C    �  �  @  �  g   �  �  �  �  �  �  �  �  �  �  �  �  �  g  D  !  �  �  �  q  <  2  !    �  �  �  �  �  �  �  q  T  8     �  '  �  �  I    �  �  �  �  �  �  �  �  �  �  �  �  �  c  F  ;    �  �  "  �  �  �  �  �  �  �  �  �  S    �  �  8  �  �  &  �  +  0  3  	  �  �  {  E    �  �  e  *  �  �  �  ]  1    �  �  x  m  �  �  �  �        "  '  ,  1  .  ,  -  1  5  6  6  6  w  b  8    �  �  ]     �  �  |  A  �  �  J  �  v    �   �  �  x  j  Z  I  8  $    �  �  �  �  i  >    �  x  B  '    S  I  @  7  -  "      �  �  �  �  �  u  _  L  D  ?  E  L  �  �  �  �  �  �  ~  v  o  h  a  Y  R  I  >  3  &    �  �  k  x      {  r  i  `  U  J  >  2  &    �  �  �  [  �  *  �  �  �  �  �  �  �  �  �  �  �  f  E  "  �  �  �  �  P      	      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  W    �  �  O  N  A  3  )    �  �  �  �  H    �  r    �  4  �  b  *  e  W  =  %    �  �  �  �  �  �  �  v  #  �  �  T  �  z  <  W  V  U  Q  F  :  .         �  �  �  �  �  x  X  A  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  j  _  �  �  �  �  e  9    3  E  E  (  �  �  ~  -  �  Y  ,    e  �  �  �  �  �  �  n  U  :    �  �  �  t  4  �  |    �  ,  >  i  �  �  �  �  �  �  �  q  X  9    �  �  �  C    �  {  0  *  "        8  ;  2    �  �  �  ~  K    �  �  W  R  :  O  ^  j  r  p  h  _  R  @  *    �  �  �  ~  C  �  �  _  �  �  �  �  b  L  Q  J  <  1      �  �  �  �  �  �  �  �  %  ]  �  �  �  �  �  �  �  �  �  �  O    �  !  �  �  �  {  �  �  �  �  {  m  U  =  %    �  �  �  �  �  �  s  ]  G  0  @  6  -  #      �  �  �  �  �  �  q  W  :       �   �   �  �  �  �  �  �  �  �  b  '  �  �  P    �  �  %  �  �  L  �  �  �  �  �  �  d  <    �  �  j  N  6      �  �  :  �  �  �  �  �  �  �  �  �  �  �  �  �  |  u  e  D  #    �  �  �  f  o  y  z  z  w  k  U  4  	  �  �  \    �  2  �  3  c  K  �  �  �  �  �  �  �  x  _  @    �  �  7  �  j  �  g  �    �  �  �  �  �              �  �  N  �  �  �  W  �  q  �  �  |  p  a  P  >  +      �  �  �  �  f  8    �  j  �  �  
  7  O  ^  e  d  R  +  �  �  �  ?  �  �    �  �  �  t  �  �    }  s  d  N  4  0    �  �  �  |  W  ?  :  �  �  �  (    �  �  �  �  �        �  �  �  �  �  �  �  �  s  a  �  �  �  �  �  �  �  x  k  b  V  E  2      �           �  |  p  d  X  L  9  $  
  �  �  �  �  �  �  �  �  �  �    �    0  V  v  �  �  �  �  �  �  j  .  �  �  ,  �  0  �    L  M  L  H  >  -    �  �  �  �  d  '  �  �  )  �  %  }  '  �  �  �  |  P  '     �  �  �  W  +  �  �  �  :  �  ^  �  �  &          �  �  �  �  M  &    �  �  1  �  s    �  �  �  �  �  �  �  �  �  p  R  3      �  �  �  �  �  �    s  [  C  *    �  �  �  �  �  �  i  M  1    �  �  �  �  |  ]  1  ,  '  '  2  <  A  C  D  @  <  8  8  8  5  -  %        �  �  �  �  �  �  �  �  {  W  (  �  �  _    �  �  *  �  h  �  �  �  �  �  �  t  V  1    �  �  u  @    �  �  c  6  "  Q  e  �  �  �  �  �  �  �  k  L  (    �  �  w  F    �  �  �  �  �  �  �  �  �  �  �  �  v  e  R  =  (     �   �   �   �  �  �  v  Z  >  $    �  �  �  �  �  �  v  ^  C  (     �   �  �  �  �  �  �  �  �  �  �  �  n  :  �  �  6  �  _  �  �  %  �  �  �  �    z  u  r  p  n  j  d  ]  Y  Z  \  ]  ^  _  `  	�  	�  	o  	:  �  �  �  =  �  �  $  �    �  �  j    �  �  �  :  =  >  9  0  &        �  �  �  �  �  v  H    �  �  m  �  �  �  �  �  �  �  �  �  �  �  �  �  k  Q  4    �  �  �  /  	  �  �  0  
�  
�  
D  	�  	�  	)  �  V  �  _  �  I  �  *  Q