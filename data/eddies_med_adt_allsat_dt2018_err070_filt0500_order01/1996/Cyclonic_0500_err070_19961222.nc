CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��1&�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M뛢   max       Pt$�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��G�   max       <o       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Q��R   max       @F(�\       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�\(�       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q            �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �hs   max       �o       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�N;   max       B/��       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B0>�       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�l]   max       C���       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��'   max       C��       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          f       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M뛢   max       PY6�       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�� ě��   max       ?�����       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��S�   max       ;��
       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?aG�z�   max       @F(�\       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v��\(��       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��`           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C   max         C       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0�   max       ?��J�M     0  ^               %                        $         #         6      &      8               =          <                                 *      f      	                  '                        
               $      )            P  N7�[O
��N�RO�l�N�9N5��N|VOx<�O��&N�
KNA��O�)�OE|�N���O�n[Nż�N&a$Pt$�N�[�PV�N�w�PS�<N��AM뛢N���N2�9PY6�N���OV��P?S�Ov�cOG}�N�G�O{��N�O��Ng��O��N�Q&OW�O�nMOB�O�c�N��.Nk�N�[N�h�N���N��O��O�4LN��O�@�ORH�N�%O�'Oq��N�_Ou�OoH&P�UN�YO �O�k�O���O��O�,N�˓Nn6iO;U�<o;��
;�o��o���
�ě��o�#�
�D���T���T���T����o��C���C���C���C����㼣�
���
��j���ͼ�����/��/��`B��`B��h�o�+�+�+�C��t��t��t��t�����w��w�#�
�#�
�'''',1�,1�8Q�D���P�`�P�`�P�`�Y��]/�]/�aG��e`B�q���q���y�#��o��o��\)���w���w���w��E���E���^5��G��������	����������"adnz~�������zqngba`a��������������������mrz������������zplkm��������������������KN[gtxtlg[RNKKKKKKKK�&'��������,7;BN[^immj[NB5,00-,�����	�������IO[hnqhf[OEDIIIIIIIIW[htuxtsjhc]d[WWWWWW��������������������)/HVag[afaZUHA>734/)SUWZanz�|~znaUSSSSSS����������������������������������0/##025000000000���������������}��'*6CDFC?6*���������������������������������������F^tw���������zaTH=<FMOQ[hokhfhihh[SOQOMMehjtywthf_eeeeeeeeee46BIOS[\[ONBBB@;9644{�����������{{{{{{{{�#0IUn{���{U<
��]gt�������tlgb]]]]]]AHUasz����zwnfaUHEAA������
��������ahkt���������thebc^a���������������������������������������������

��������������������������������%% ��������w{���������~{uwwwwww!'0<@IMMJF@<20#!$# !)0<?INONIC<;20*$))))lmxz������������zmll��������� �����������)058;885���
!#/9:96.#
�����wz�����������}zyxvww��������������������')5BFKNPONNB=50)&#''������������������xz�����������}zwxxxx��������������������	#/<HU`blUG</#	&)6B[hrvropg[OJB3(&&ghot��������ttlhggggMPXg�����������tg[NM()5BN[_fdd^UNBA;4-*(BBMO[`htxtslh_[OFBBB)-59;BNQQNLNPPLB50))���������������������������������������������

�������wz���������������zoweuz��������������|de����������������������������������������ot������������tolklo������������������������

�����������	������ ///<FHJHH?<7//////// #*/<HU\[UTSMH@</# �4�(������4�A�M�Z�s�~��������f�A�4�ֺ˺Һֺ������ֺֺֺֺֺֺֺֺֺ��L�A�8�<�A�N�V�Z�g�s�~�������~�s�g�c�Z�LŠŚŞŠŭŹ��������ŹŭŠŠŠŠŠŠŠŠ����û����������������$�#�&�%������������¿´²®²¿���������������������ؽ���������������������������������A�8�4�,�+�4�A�E�L�M�M�M�A�A�A�A�A�A�A�A���s�g�e�p�~�����������������������������	���׾˾ʾ׾�	��"�.�3�:�=�8�.�"��	����������������������������������������ÓÑÏÓÓàêìùûùìàÚÓÓÓÓÓÓ�@�8�<�F�J�I�N�e�r�������������~�r�e�L�@�
���
��#�<�G�H�K�U�W�Y�U�H�<�#���
����������������������������������������ÐÇ�z�u�p�k�zÃÇÓìíùÿ����ùìàÐ�M�C�A�?�A�M�Z�f�s�����������s�f�Z�M�M�.�.�*�#�!������!�.�.�.�.�.�.�.�.�.�y�`�H�@�F�;�G�m�����ѿݿ�������ѿ��y�"�!���
���"�*�.�1�2�4�.�"�"�"�"�"�"��	������"�/�;�T�[�f�t�v�m�a�\�H�/�"��U�S�L�K�U�`�a�n�s�zÁÇÇÇ�z�n�m�a�U�U����ѿĿ��������ݿ���$�5�N�����w�Z�5��Ϲʹù����ùϹٹܹ����������ܹ׹ϹϺ�����'�)�(�'������������l�f�f�l�p�x���������������������x�o�l�l���������������������������������������������������q�i�b�a�f�m�~������������� ��������������	�������	�������������A�8�3�7�A�F�M�Z�f�����x�s�q�j�_�Z�M�A�λ��ûܻ������Y�h�k�e�Y�@�����ν��������������������нܽ���ݽнĽ����������������������	��"�+�"��	��������������������$�+�0�2�0�/�$�$������������öïùû�������������� �������Ҽr�k�r�w�������������������r�r�r�r�r�r�ʼ����ʼ׼����!�5�@�E�E�:�!�����ʻS�H�M�S�_�l�x�������x�l�i�_�S�S�S�S�S�S�������������������Ľнֽݽ��ݽнϽĽ����������(�3�4�5�4�(�#���������������������������
������
������ĕĂ�t�`�c�h�tāčĦĿ��������������ĿĕŠŕŔŋňŎŔŜŠŭŹ����������ŹŵŭŠD�D�D�D�D�D�EEEE*E7ECELENEJE@E*EED�F$F!F$F$F,F1F=FJFVF^FcFeFcFVFPFJF=F1F$F$�ʼ����������ʼмּܼ���ּʼʼʼʼʼ������������������������������������������b�^�I�=�;�=�@�I�P�V�b�m�o�{�ǁ�{�o�m�b����� ����!�!�!�&�)�%�!�������~�z�r�e�Y�V�Y�e�j�r�~�����������������~���������������������ſѿ׿���ݿѿĿ����y�u�z��������������������������������	���������� �	���"�#�"�"���	�	�	�	������������������1�5�B�N�L�D�9�)��#�"�����#�/�<�H�U�a�l�d�a�U�H�<�/�#�	���������������	�������	�	���������ùïù��������������� ������
��!�-�:�F�S�_�l�������x�k�_�J�#��l�f�_�S�M�K�F�@�F�S�_�l�p�q�s�m�l�l�l�l����������������!�+�-�8�:�=�:�-�!��H�D�;�4�/�,�/�3�>�H�T�a�l�o�p�m�h�\�T�H��²�r�o�t²����������#�%������/�)�/�4�<�H�T�H�H�<�/�/�/�/�/�/�/�/�/�/�Ŀ������Ŀʿѿݿ�����������ݿѿȿĿ������������������#�0�8�A�@�9�0�#�
�������������*�6�C�O�P�Q�O�K�B�6�*����������������������
��� �#�"���
����l�h�`�S�F�G�S�`�y�������������������y�l�!���������!�.�:�=�:�5�.�!�!�!�!D�D�D�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� 2 1 @ O 3 > u f 1 T I O J F L ; f G N X ! ' y o P Y \ b X P T - i O $ ~ [ h D < ; O # / = E = 6 P ] _ B ) ; , h � � Q ( A w c R ' ; % � _ D =    A  P  1  �  �    �  �  �  j  �  u  )  �  �  H    X  ^  �  L  �  \  �    �  }  S  �  �  }  �  G    �  �  L  �  ]  �  8  {  �  �    �    !      i    �  �  �  �  �  �  �  -  �  L  ;  <  e  T  ;  �  �  d  ���o�o�T���ě��'e`B�#�
��o��/�49X���
��C��T����h���
�T����9X��j�����/�u�8Q콧#�
�C��\)�o��Q��w��o��vɽaG��<j�D���y�#�#�
����,1�P�`�@��L�ͽ��o�hs�]/�H�9�e`B�ixս�hs�]/���
��Q�m�h��O߽��w�m�h������C���+��O߽��T�\��\)������l���/��F���ͽ��`��/�bNB�Ba�BnBmB ��B�B�yB�|BHB�B1vB�BB!��BDB�1B��B#�B%P�B+A�B/��B��B �=A�N;Bh�B��BXB<uB&9�B	��B+�B�fB��BQUBO�B�B+g�B-e�B)bB&:B&+B U�B2hB<9B�LB�%B .�Bv�B
��B�By�B�mB��B�OB	��B��BB}B�sBB"-
B#�_B�kB=B
�yB�B
�BęB��B�BX�B�rB��B��B�kB8�B��B ��B<�B	6B�BmB�XB	�B�"B!�UB@B��B�B"�9B%AsB+:�B0>�B1�B �A��B@3B�BA�B��B&��B	��B��B�*B�+B�2B�BAdB+�B-��B(�0B%�B&�TB bB?�B��B��B��B ;�BCB
��B)�B��B�B��B�IB
�BݼBt�B��B�QB"J�B#��B�B?2BZB�B
LkB��B�B��B@ B�SB�'A>�@B�A���A� AҗA��A.�|A:E6A�1�AZ��A�7~Aˤ�?��iA�:A��A�1�AA2A��As[jA_)�A���A��A��A>�l]?��@���@���A��XAZP�A=7�@Ķ�A$�}A���B	1�A���@�^A�(@�	A$��A4]!A�[�A�N�A�6�C�r~C���@��A�X�B0A
�2@�<AwA�XyA��?A��eAò�AY��AҌ�@��@��u@d�A�4zA�K*Aå2A|��A�ZA�$�A�j�A��A�*C��C�]�A<��@C�~A�1ZA���A҄\A�vtA._/A:�[A��VA\uzA���Á&?�A���A��A�~�A?�A�Ao3'A^�3A���Aƀ"A��">��'?��6@� '@��dA�s�A[ A<�Y@�A#�A�J�B	��A���@�}A	 r@��A#�A5 A�r�A��A���C�x�C��@��A��B��A
��@~Ax�MA��)A�A�g�AÀAZZAҏ�@�j�@�b�@k�AA��kA�.A���A}4A腫A�VA�i�A ��A C��(C�W�               %   	                      %         $         6      &      9               >          =                           	      +      f      	                  (                                        $      *                %                                    %                  7      %      7               9         1                  )               %      #                        #      !            !            1                                                                                    1      #      /               9         -                  )               #                              !      !            !            +                           O�]#N7�[O
��N�RO�ّN�9N5��N|VOx<�O�x>N:U�NA��O���OE|�N���O`��Nż�N&a$PJ�XN�[�O�R�N�w�Pd�N��AM뛢N���N2�9PY6�N���NP+��Oi>O0�N�G�OmiPN�O��Ng��N�N7N�Q&OW�O�B�OQ�O���N�lvNEKbN�[N�h�N���N��O��O�X�N��O�@�O(�N�%N��8Oq��N�_N�c�O*bO�ON�YO �O�m�O���OuW�O�,N�˓Nn6iO0ʿ  #  �  �  �  �  �  m  A  Z  �  �  �  �  |  F  	    )  &  �  �  e  O    �  �  "  �  	  �  �  4  t  0  �  F    �  ?  �  @    �    �  �  y  �  �  �  ?    x  &  S  �  �  �  F  R  �  �  n  �  P  �  �  	  �  �  p:�o;��
;�o��o�#�
�ě��o�#�
�D���u�u�T����j��C���C��ě���C������/���
��h���ͽ�w��/��/��`B��`B��h�o�#�
��P��w�\)�t���P�t��t����#�
��w�#�
�49X�49X����0 Ž,1�,1�,1�<j�D���P�`�aG��P�`�Y��m�h�]/�u�e`B�q���u�����7L��o��\)���T���w���E���E���^5��S���������������������"adnz~�������zqngba`a��������������������z�����������ztonmpuz��������������������KN[gtxtlg[RNKKKKKKKK�&'��������,7;BN[^immj[NB5,00-,������������KO[hjmh[OGKKKKKKKKKKW[htuxtsjhc]d[WWWWWW��������������������)/HVag[afaZUHA>734/)SUWZanz�|~znaUSSSSSS�������
	�����������������������������0/##025000000000��������������������'*6CDFC?6*����������������������������������������Nmz���������taTHDBCNMOQ[hokhfhihh[SOQOMMehjtywthf_eeeeeeeeee46BIOS[\[ONBBB@;9644{�����������{{{{{{{{�#0IUn{���{U<
��]gt�������tlgb]]]]]]DHUalnz}zzrna[UIHDDD��������������rt|����������}tmmlmr���������������������������������������������

��������������������������������%% ��������w{���������~{uwwwwww"#(0<ILLIIE?<0'#!!"")0<?INONIC<;20*$))))lmxz������������zmll�������������������� )-5753*) ��
#&+,(#
������yz���������zzzyyyyyy��������������������')5BFKNPONNB=50)&#''������������������xz���������~zwxxxxxx��������������������	#/<HU`blUG</#	()6B[hmrrlhe[OFB:*((ghot��������ttlhggggMPXg�����������tg[NM15BNS[```[ZQNIB>60-1BBMO[`htxtslh_[OFBBB056=ABBJNNIDB655,,00����������������������������������������������

�����������������������~}�glrwy������������thg����������������������������������������qt������������tqnlnq��������������������������
����������	������ ///<FHJHH?<7//////// #$*/<HVZUTRLH></# �Z�M�A�8�3�%�!�&�1�4�A�M�Z�r������s�f�Z�ֺ˺Һֺ������ֺֺֺֺֺֺֺֺֺ��L�A�8�<�A�N�V�Z�g�s�~�������~�s�g�c�Z�LŠŚŞŠŭŹ��������ŹŭŠŠŠŠŠŠŠŠ�������������������"�!�����������������¿´²®²¿���������������������ؽ���������������������������������A�8�4�,�+�4�A�E�L�M�M�M�A�A�A�A�A�A�A�A���s�g�e�p�~�����������������������������	�����׾о;׾�	��"�.�1�9�;�5�.��	����������������������������������������ÓÑÏÓÓàêìùûùìàÚÓÓÓÓÓÓ�Y�P�P�M�M�O�W�e�r�~���������������~�r�Y�
���
��#�<�G�H�K�U�W�Y�U�H�<�#���
����������������������������������������àÓÉÇ�{�v�tÁÇÓàìóùþþùõìà�M�C�A�?�A�M�Z�f�s�����������s�f�Z�M�M�.�.�*�#�!������!�.�.�.�.�.�.�.�.�.���m�V�L�F�D�G�T�`�����Ŀѿ�� ����ݿ����"�!���
���"�*�.�1�2�4�.�"�"�"�"�"�"�/�"�������"�;�H�T�a�o�o�h�^�T�H�/�U�S�L�K�U�`�a�n�s�zÁÇÇÇ�z�n�m�a�U�U����ݿпʿɿͿݿ����5�N�r�w�d�Z�N�5��Ϲʹù����ùϹٹܹ����������ܹ׹ϹϺ�����'�)�(�'������������l�f�f�l�p�x���������������������x�o�l�l���������������������������������������������������q�i�b�a�f�m�~������������� ��������������	�������	�������������A�<�6�:�A�C�L�M�Y�Z�f�h�i�f�e�Z�Y�M�A�A�ѻǻĻ������4�Y�e�i�d�Y�S�@�����ѽ����������������ĽнԽݽ޽ݽڽнĽ����������������������������	��� ��	������������������$�+�0�2�0�/�$�$������������÷óùý�����������
�����������Ҽr�k�r�w�������������������r�r�r�r�r�r�ʼ����ʼ׼����!�5�@�E�E�:�!�����ʻS�H�M�S�_�l�x�������x�l�i�_�S�S�S�S�S�S���������������������ĽнԽ۽н̽Ľ��������������(�3�4�5�4�(�#���������������������������
������
������ĿĦĘĆ�t�d�h�lāčĚĦĳĿ����������ĿŠřŔōŊŐŔŠŭŴŹ����������ŹŰŭŠD�D�D�D�D�D�EEEE*E7ECEHEEE9E*EEED�F1F+F'F/F1F=FJFVFZF`FVFJFIF=F1F1F1F1F1F1�ʼü��������ʼμּۼ߼ּʼʼʼʼʼʼʼ������������������������������������������b�^�I�=�;�=�@�I�P�V�b�m�o�{�ǁ�{�o�m�b���������!�&�(�$�!���������~�z�r�e�Y�V�Y�e�j�r�~�����������������~���������������������ſѿ׿���ݿѿĿ����~�x�|���������������������������������	���������� �	���"�#�"�"���	�	�	�	������������������1�5�B�N�L�D�9�)��#�"���#�)�/�<�H�T�U�a�e�a�^�U�H�<�/�#�	���������������	�������	�	�����������������������������������
��!�-�:�F�S�_�l�������x�k�_�J�#��l�f�_�S�M�K�F�@�F�S�_�l�p�q�s�m�l�l�l�l��������������� �!�*�-�4�-�,�!���;�2�3�7�;�C�H�S�T�W�a�g�j�j�f�a�]�T�H�;����¿²�v�r�v¦¿�������	������/�)�/�4�<�H�T�H�H�<�/�/�/�/�/�/�/�/�/�/�Ŀ������Ŀʿѿݿ�����������ݿѿȿĿ����������������
��#�0�3�<�=�6�0�#�
�������������*�6�C�O�P�Q�O�K�B�6�*��������������������������
���!� ��
����l�h�`�S�F�G�S�`�y�������������������y�l�!���������!�.�:�=�:�5�.�!�!�!�!D�D�D�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� : 1 @ O - > u f 1 W M O 3 F L @ f G J X % ' z o P Y \ b X B S 4 f O # ~ [ h 2 < ; F  3 6 G = 6 G ] _ = ) ; + h b � Q & E r c R ( ;  � _ D 9    h  P  1  �  /    �  �  �  8  h  u  +  �  �  �    X  �  �  �  �  �  �    �  }  S  �  $  S  G  �    �  �  L  �  	  �  8  �  K    �  _    !  �    i  �  �  �  \  �    �  �  �  �  �  ;  <  '  T  �  �  �  d  s  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  �  �        "         �  �  �  �  S    �  �  �  c    �  �  �  �  �  �  w  d  Q  ;  %    �  �  �  �  �  M  �  �  �  �  �  �  �  �  �  �  t  V  7    �  �  �  w  H    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  k  `  U  J  �  �  �  �  �  �  �  �  Y    �  �  M  �  �    �    �  z  �  �  �  �  �  �  o  X  A  )    �  �  �  �  K  
  �  P   �  m  q  t  w  {  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  A  7  -  #      �  �  �  �  �  �  m  P  2     �   �   �   �  Z  V  O  F  <  6  8  6  ,      �  �  �  �  q  O  /      �  �  �  �  �  �  �  �  �  r  Y  7  
  �  s    �  �  e  �  �  �  �  �  �  �  �  �  �  �  x  p  c  U  C  )    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  w  r  l  e  ^  X  Q  �  �  �  �  �  �  �  �  �  z  Z  8    �  �  x    �  d  �  |  k  ]  T  F  9  3  3  6  7  4  -  $      �  {  ?    �  F  ?  9  2  +  %                �  �  �  �  �  �  �  �  �          �  �  �  �  m  9     �  �  2  �  j  �      �  �  �  �  �  �  �  �  �  �  �  �  �  y  o  c  V  I  <  )          �  �  �  �  �  �  �  �  �  �  �  �  �  {  m      $  $    �  �  �  �  W  $  �  �  f    �    w  �   �  �  �  w  k  h  f  X  A  *    �  �  �  �  �  �  �  d  ?    g  �  �  �  �  �  s  [  =    �  �  s  +  �  �  8  �  n  �  e  4    �  �  _  $  �  �  �  �  4  �  z  )  �    $   �   g  %  +  9  B  L  N  G  4    �  �  }        �    e  u        �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  -  �  q  �  �  �  �  �  �  �  �  �  �  r  Z  B  *    �  �  �  h  <  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     �  �  "      �  �  �  �  �  �  �  w  d  R  ?  -  �  �  �  J    �  �  W    �  �  �  P    �  �  �  f  1  �  �  I  �     �  	    �  �  �  �  �  �  �  �  �  �  v  `  I  0    �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  -  �  �    �    �  2  v  �  �    n  G    �  �  `  T  ;  �  �  0  �  A  �    f      '  /  3  4  0  '      �  �  �  �  p  G    �  �  �  h  p  q  k  d  \  Q  D  7  (    	  �  �  �  �  �  j    �  0  +  %         �  �  �  �  �  �  f  B    �  �  �  r  3  �  �  �  �  �  r  ^  F  (    �  �  \    �  n  �  S  �  C  F  ;  0  %      
    �  �  �  �  �  �  �  r  Y  A  (      �  �  �  �  ]  4    �  �  \  2    �  �  �  `    �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  p  k  f  `  [    *  =  9  4  +      �  �  �  �  �  �  h  G  $  �  �  �  �  �  ~  x  r  l  f  `  Y  Q  H  >  4  +  "        "  -  @  6  +                �  �  �  �  �  �  �  Z    �  �  �    �  �  �  �  �  �  �  �  �  Y  !  �  �  �  +  @  O  �  �  �  �  �  �  �  �  s  M  '  �  �  �  [    �  �  t  �  +  {  �          �  �  -  �    _  �  �  �  
7  �  v  �  �  �  �  �  �  �  �  �  �  �  �  p  V  <       �  �  j    ~  �  �  �  �  �  }  y  z  {  n  Q  5    �  �  �  �  f  A  y  o  b  O  5    �  �  �  �  q  Q  )    �  �  �  �  �  �  �  �  �  �  �  �  v  Z  ;    �  �  �  �  e  ?  	  �  �  �  d  ~  u  l  c  W  @  (    �  �  �  X     �  �  ?  �  �  
  �  �  �  �  �  �  �  t  b  O  ;  '    �  �  �  �  D   �   �  ?  2  #    �  �  �  �  `  0  1    �  �  [    �  
  `  �  �  �    �  �  �  �  �  b  ,  �  �  t  5  �  �  "  �  �  4  x  p  g  \  L  ;  )      �  �  �  �  �  �  p  X  ?  %    &      �  �  �  �  �  �  t  R  -    �  �  ~  \  @    �  Q  R  Q  R  Q  P  M  E  5       �  �    P  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  f  V  F  6  &  :      �  �  �  }  O    �  �  q  8    �  u    `  �  �  �  �  �  �  �  t  R  =  -    �  �  �  �  `  5    �  �  {  F  :  -          �  �  �  �  �  �  p  \  L  A  5  ?  R  d  :  G  Q  L  E  6  %    �  �  �  �  �  j  O  3    �  �  >  �  �  �  �  �  �  �  �  �  �  �  �  d  4    �  �  �  �  �  �  �  �  �  �  c  *  �  �  j  2  �  �  �  F    �  Z  �   �  n    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    j  Q  4    �  �  �  {  I    �  s    �  T  �  �  �  D  O  O  L  E  9  '    �  �  �  �  ?  �  �  &  �  D  �  �  �  �  �  �  g  @  *  "        �  �  �  ]    �  v  *  �  �  �  �  �  �  |  Y  .  �  �  �  H  �  �  =  �  i  �  �  '  	  �  �  �  �  {  a  I  0  /  V  L  &  �  �  _    �  ?   �  �  |  e  F  $    �  �  �  �  h  >    �  �  �  �  g  B    �  �  �  )  �  �  V    �  ]    �  \    �  H  �  �    �  j  d  ?    �  �  �  \  !  �  �  >  �  �  #  �  _    !  