CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�&�x���     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��7   max       P�0g     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��Q�   max       <�C�     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?:�G�{   max       @F��\(��     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vy\(�     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�c@         $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��l�   max       $�       $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�FS   max       B4��     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�xS   max       B4�G     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =K��   max       C��V     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       :��7   max       C��`     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          C     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��7   max       P�0g     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���m\��   max       ?��x���     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��Q�   max       ;��
     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?:�G�{   max       @F��\(��     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vy\(�     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�c@         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F/   max         F/     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?��x���     �  _�   ,            A            ,   	   	   )            <         C                                             %            *   
                               
            	            
                                                PM��NL�YNڐeN#0P"a�Oz��N^�O
�P�Nb�wN19$P�0gN���N�Ne(cP#�N�^�N��aP26N}��O�N��tN��OZ�O���N���O�<NpZO�}O`~;NW�OK�)NF<P��P$3�M��7N�d�P
#�N���O�ϩO�~N���O1$@Og�N2N
TN���NW�:N���N[�N�@N?�N���O%��Nj�dN��NW^�O��JO@��O��OpANj�LN�wO�2N�O�dN��xO��;O+�:N��^O aZN��<N��a<�C�;��
;�o:�o��o��o�ě��o�o�t��D���T���e`B�e`B�u�u��C���t���t����㼛�㼣�
��9X��9X�ě��ě����ͼ��ͼ��ͼ�������������`B��h��h���C��C��\)�\)�t��t��t���P��P��P���#�
�#�
�#�
�'''49X�8Q�@��D���D���P�`�]/�aG��e`B�ixսm�h�u�u�y�#��o���P���P����������Q�"6B[x���lh[6)��������������������,/<HPUZ_ZUH?<3//,,,,��������������������Za�������������zpnYZ������� 
������NNR[agigf\[VNJNNNNNNyz~�������������ytty*CO\l~��zuhO*�����������������������������������������
#<bn{������U<0
����� �����������=IJUbkib`VUIB@======)5@BBB5)��������������������<<CHUXX[\UHA=<<<<<<<������������������������������������������������������#/9<HSLHG=</##����������������������������������������
#&0#
����������������������������������������������FHMTaz�zxxxwmaTMHFF����������������������������������������������
��������FHJSUaha_][UHHFFFFFFLOOX[hpt������th]SOL������ ��������������BNeq[UQB5)�����)O[fljVOB)������������������������_bnw{���������{nnlh_�����$$���������))15>BCEIKIBA5)'%'))����
#.#
�������������"�������.6BOSZ[^[WOB>760.....2<FIJUXXVRKI<60-,,.����

��������RUadglaUTSRRRRRRRRRR��������������������jnz�������zwqnjjjjjj����������������������������������������/039<HIUQI<0////////LOQ[httutptuth[XONLL����	�������������"#05<GE<;0#����������������������������������������;HTWTTOIH;535;;;;;;;bgt����tg`bbbbbbbbbb�������������������������������������������!�������
 )076)���6<HPUX^aekaUNHB<:766uz~�������������xrpu")*./)	x����������������}xx��

����������5BMTH<)	��������
��������knz���������znfdkkkk)5A750)#mtu����������ttnmmmm4<HHNQPHD<7112444444�`�W�;�*�)�%�;�`�m�y�����������������y�`����������#�������������4�,�,�/�4�9�A�M�Z�_�`�Z�W�M�B�A�4�4�4�4�����������ʾϾʾʾ���������������������������ĿķĵĿ���������"�0�8�;�0�#�����#� ����$�2�<�H�U�a�n�v�w�n�\�U�G�<�#�f�f�Z�R�Z�b�f�p�s�����|�s�f�f�f�f�f�f�I�G�I�L�U�\�b�j�n�{ŇŔŏŇŅ�{�n�b�U�I����Ѿ����þ׾�����.�7�<�=�6�.�$��������������������
����������������FFFFFF$F'F)F$F#FFFFFFFFFF���������s�R�?�/�'�-�I�������������������t�r�t�v¤�t�t�t�t�t�t�ʼ����������ʼԼּ����ּʼʼʼʼʼ��I�G�F�F�F�I�T�V�V�Z�]�Z�V�Q�I�I�I�I�I�I���ܻû������������лܻ������.�2�(��������������������������������������������O�L�F�E�O�\�h�h�l�h�f�\�O�O�O�O�O�O�O�O�����}�z�x�}�����������ּ�������ּʼ������#�$�/�0�=�B�=�<�3�0�&�$�����H�A�B�H�J�P�U�a�j�n�z�|��z�z�n�a�U�H�H�����������ûлܻ���ܻлû������������H�A�?�F�H�T�a�l�g�l�a�T�H�H�H�H�H�H�H�H�H�A�3�+�-�4�@�N�Y�f�r�~�����}�r�f�Y�H��žŵŵŹ������*�6�O�\�O�I�6�*��������Z�T�Z�\�g�q�s�u�������������s�g�Z�Z�Z�Z����ƳƦƝƟƧƳ����������������������ƎƋƁ�{ƁƎƚƧƭƬƧƚƎƎƎƎƎƎƎƎ�������������������¿Ŀɿܿ��׿ѿĿ���������׾˾ɾ׾���	��"�+�/�/�.�"���H�>�;�/�-�*�/�6�;�H�T�[�U�T�H�H�H�H�H�H�ݽ׽нĽ����������Ľнݽ���������߽ݺɺú��������ɺ˺ֺ޺ֺҺɺɺɺɺɺɺɺ�ŠŎŇņœŕŠŭ������������������ŹŭŠ��������������������������� �����������*�#�����#�*�1�6�8�6�*�*�*�*�*�*�*�*��������0�4�@�D�M�T�M�E�B�@�4�'������������ֽ��!�1�6�5�)�����ʼ��������������������������������������ݿͿ¿��Ϳѿӿ׿ݿ������������ݻ����������������������лܻ������лû����������������ûɻл׻һлû����������������������(�4�A�D�P�M�H�A�4�(�D�D�D�D�D�D�D�D�D�D�E EE	EEED�D�D�D��(�#�"�(�5�A�M�C�A�5�(�(�(�(�(�(�(�(�(�(������t��������������������������������g�a�[�[�^�g�r�s���������s�p�g�g�g�g�g�gŇłŇŔŕŠŭŹ��ŽŹŰŭšŠŔŇŇŇŇ�F�;�:�2�:�F�S�_�d�l�x�|���{�x�l�_�S�F�F�:�8�-�!��!�!�-�:�E�D�:�:�:�:�:�:�:�:�:�Ϲȹù����ùιϹϹܹ�������ܹܹϹϽ������������ĽͽʽĽ������������������������������������������������������������o�b�V�S�O�I�I�V�X�b�o�{ǆǈǊǊǈǀ�{�o������������������������	�������������������(�+�2�(��������������������	����	����������������������������Ҿ׾�����"�0�7�3�.�"��	���ܹڹӹֹܹ޹���������������ܽS�G�6�.���!�.�G�`�y�������������l�`�S�Z�M�A�4�(�#��(�4�H�Z����������}�s�f�Z���ݽ������������������鹝���������������ùϹعֹϹɹù����������ù����������ùϹܹ�����������ܹϹ��O�L�C�O�Q�[�f�h�tāąĆā�~�t�j�h�[�O�O������ýþ������������������������������E\E[EPECE8E?ECEPE\EgEiErEuExEuEiE\E\E\E\�����������õìðù����������%����N�L�S�Z�`�g�s���������������������s�Z�N�����������������ĿɿѿԿ׿׿ѿĿ�������������������������������� ��������������������������������
����
�	����������D�D}D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�  A F > G  > e * ] J _ : Q | ? ; 9 5 � & $ 5 F l ? O e 9 f ^  9 @ C m B T K e P < T I P T @ � t 9 / 1 F ( � O ` Z $ M Y 9 h P R . U [ p Z 4 9 =    O  Z  	      �  W  Z  �  �  b  *  �  �  �  �  �  �  �  �  6  �  �  �  �  �  <  p  E  
  }  �  <  �    5  -  �    �  F  �  �  M  S    �  �    |  �  [  �  g  �  �  �  �  �  �  C  r  !  q    @  �  �  �    .  �  ϼ���$�  �49X�D����C���/�#�
��j�T����C����
�]/�����
���㽗�P��9X���ͽ�1��j�o�C������#�
�49X�o�P�`��/�',1��`B�#�
�C�����D���o��w�����49X�ixս�C��aG���o��%��w�49X�,1�0 ŽL�ͽ<j�]/�8Q�L�ͽ�o�H�9�aG��m�h��\)���P���w���w�q����\)�����������w��������ȴ9��E���vɽ�-��l�B��B�BWB4��B@XB��B��B v�B0�WB=�B˔B&�"B�B'")B�"B WBt6B��BYiBCXBQ�B��B�B$1�B��Bn�A�ٞB$B*�+B�DB��B?B#5�B��Bl8BC!B(ӒB-�-Bl"B:�B*zBt�B&ywB"$B@B �Bz	BcB��B&m&B�3B#��B%��B@B��A�FSB	�:B<�B ޠBȣB��B7�B~|B�B	�B�B�*B�Bz�B�QBWfB
eOB�1B��B:�B;�B4�GBҾB:�B�AB �iB0B�B}�B�:B(��B� B'qB��B ?�BN`B$�B{�B��BϾB�eB}�B$?�B��BC�A���B,}B*PB�0B�{B@�B#<�B�ZB9�B<�B(��B-�|B�B�6B?�B��B&BB5�B8B��B�B��B��B&ZBHB#�2B%@�B=�BADA�xSB	��BB�B �>B?�BFyBBYB@B��B>lBI�B/�B�=BAUB@�B@>B
E�B��Am�"A2y3A;�UAO!�A��nAĤ�AA�+A�Q�AY��A�11C��VA���A�2�@�]�BY>@��0A�B�I@��B
|A�U�@���A��G@ܾ�A���A�VyBx�B#�Awe>AYs�A�
�A*�U@3��A�E+A��A��[@��AYjA�SDA��@�|@��qA6�C�6�A��@� |A���A���@���@xFr>���A%��A"nmB��A��oA� AY�AY�?7PUA@rAA1A/e�=K��>��	A�h{A�SC��[A��A�K�AxV�A�[�A�gOC��,An�A2��A;�AO	{A姂AÅ�AAlwA��QAY
YA��QC��`A��/A�~�@���B��@��9A�OnB�S@�I6B
?;AŌW@��-A�l@���A���A�}}BRpBhZAv�AZQ�A�|A+Q�@3��A��jA�o�A�{�@���A�A���A�q�@�c@���A7nC�7�A�i�@�|A� A�\�@�*@t}>�&�A%3�A"��B��A�TDA��AZ� AY/?0 �A9�AB�A/�c:��7>��A���A�bC��AϦ�A�� Ax�A�nNA�j�C��   ,            A            -   	   	   *            <         C                     	                        %            *   
      !                                    
         	                                                      -            )   !         )         E            -         '                  %                           )   3         -      !                                                      #      %                        %                              #   !                  E            #         '                  !                           )   #         '                                                            #      %                                       OF$NL�YN�bKN#0O��_Oz��N^�O
�OP_N?8N19$P�0gN�T�N�Ne(cO�{�Nt��N|clP(+N}��N�2N��gN��N"�iO­�N���O���NpZO�}O`~;NW�O6��NF<P��O�v�M��7N�d�P$N���O\��O#?�N�,*O8�Og�N2N
TN���NW�:N���N[�N�@N?�N���O%��Nj�dN��NW^�O��JO@��O��O'��Nj�LN�wN̰
N�O�dN��xO�0�O+�:N��^N�E*N��<N��a  �  i  �  �  d  <  �  �    �  u  3  �  8  �  �  8  (  	o  �  *  [  �  #     C  �  �  B    �    �  �    �  s  1    �  ?  }    	  z  �  �  )    �  ,    �  S    �  O  <  �  �  �  `  �  U      �    �  2  �  f  ��#�
;��
;D��:�o��9X��o�ě��o��h�#�
�D���T���u�e`B�u����t����㼼j���㼬1��9X��9X�o���ͼě��������ͼ��ͼ���������/��`B���C����C��\)�\)���49X��w�#�
��P��P��P���#�
�#�
�#�
�'''49X�8Q�@��D���D���P�`�]/�q���e`B�ixս}�u�u�y�#��+���P���P���-������Q�*6BHO[\fhmlg[OB72.)*��������������������-/<HOUZ^YUH@<40/----��������������������jo���������������zqj������� 
������NNR[agigf\[VNJNNNNNNyz~�������������ytty%*-6COV[^_\OKC6*������������������������������������������
#<bn{������U<0
�����������������=IJUbkib`VUIB@======)5@BBB5)��������������������EHUVWYZUHC?>EEEEEEEE������������������������������������������������������ #/6<HKHGB<:/&#  ����������������������������������������

��������������������������������������������������GINTamz~zxwwumaTNIFG����������������������������������������������
��������FHJSUaha_][UHHFFFFFF[ht���������th`[VPQ[������ ��������������BNclfTPB5)������6B[fcQOB6)����������������������_bnw{���������{nnlh_�����##��������))15>BCEIKIBA5)'%'))������
��������������
�����������369BOPX[\[TOBB963333.0049<IRUVVTPII<0/..����

��������RUadglaUTSRRRRRRRRRR��������������������jnz�������zwqnjjjjjj����������������������������������������/039<HIUQI<0////////LOQ[httutptuth[XONLL����	�������������"#05<GE<;0#����������������������������������������;HTWTTOIH;535;;;;;;;bgt����tg`bbbbbbbbbb�������������������������������������������!������
&)-11-)	 6<HPUX^aekaUNHB<:766yz������������zvtyy")*./)	x����������������}xx��

�������5BKRLG;)����������
��������knz���������znfdkkkk	)5754/) 
				mtu����������ttnmmmm4<HHNQPHD<7112444444�`�\�_�`�a�`�`�m�o�y���������������y�m�`����������#�������������4�0�-�0�4�:�A�M�Z�^�_�Z�V�M�A�=�4�4�4�4�����������ʾϾʾʾ�������������������������������Ŀ����������� �������
���#� ����$�2�<�H�U�a�n�v�w�n�\�U�G�<�#�f�f�Z�R�Z�b�f�p�s�����|�s�f�f�f�f�f�f�I�G�I�L�U�\�b�j�n�{ŇŔŏŇŅ�{�n�b�U�I�	������۾վ۾����	�� �"�$�$� ��	����������� ��	��������������������FFFFFF$F'F)F$F#FFFFFFFFFF���������s�R�?�/�'�-�I�������������������t�s�t�v£�t�t�t�t�t�t�ʼ����������ʼԼּ����ּʼʼʼʼʼ��I�G�F�F�F�I�T�V�V�Z�]�Z�V�Q�I�I�I�I�I�I�лû��������������ûܻ�����!����ܻ������������������������������������������\�Q�O�G�F�O�\�f�h�k�h�c�\�\�\�\�\�\�\�\�ʼ������|�{������������������ּ������#�$�/�0�=�B�=�<�3�0�&�$�����H�D�D�H�M�S�U�\�a�n�w�z�~�z�w�n�a�U�H�H�����������ûлܻ߻��ܻлû������������H�A�?�F�H�T�a�l�g�l�a�T�H�H�H�H�H�H�H�H�Y�W�O�Y�c�f�h�r�v�r�h�f�Y�Y�Y�Y�Y�Y�Y�Y��ſŶŷŹ���������*�6�O�F�6�*��������Z�T�Z�\�g�q�s�u�������������s�g�Z�Z�Z�Z����ƳƧƞƟƦƳƻ��������������������ƎƋƁ�{ƁƎƚƧƭƬƧƚƎƎƎƎƎƎƎƎ�������������������¿Ŀɿܿ��׿ѿĿ���������׾˾ɾ׾���	��"�+�/�/�.�"���H�>�;�/�-�*�/�6�;�H�T�[�U�T�H�H�H�H�H�H�ý������Ľͽнݽ��������������ݽнúɺú��������ɺ˺ֺ޺ֺҺɺɺɺɺɺɺɺ�ŠŎňņŔŖŠŭŹ����������������ŹŭŠ�����������������������������������������*�#�����#�*�1�6�8�6�*�*�*�*�*�*�*�*��������0�4�@�D�M�T�M�E�B�@�4�'������������ּ����!�/�5�4�/�(����㼱������������������������������������ݿпſÿ˿ѿݿ��������������㻞�������������ûлܻ�����ܻлû����������������������ûŻлջллû����������(����������	���(�4�A�B�M�B�A�4�(D�D�D�D�D�D�D�D�D�D�E EE	EEED�D�D�D��(�#�"�(�5�A�M�C�A�5�(�(�(�(�(�(�(�(�(�(������t��������������������������������g�a�[�[�^�g�r�s���������s�p�g�g�g�g�g�gŇłŇŔŕŠŭŹ��ŽŹŰŭšŠŔŇŇŇŇ�F�;�:�2�:�F�S�_�d�l�x�|���{�x�l�_�S�F�F�:�8�-�!��!�!�-�:�E�D�:�:�:�:�:�:�:�:�:�Ϲȹù����ùιϹϹܹ�������ܹܹϹϽ������������ĽͽʽĽ������������������������������������������������������������o�b�V�S�O�I�I�V�X�b�o�{ǆǈǊǊǈǀ�{�o������������������������	�������������������(�+�2�(��������������������	����	����������������������������Ҿ׾�����"�0�7�3�.�"��	���ܹڹӹֹܹ޹���������������ܽS�G�6�.���!�.�G�`�y�������������l�`�S�Z�U�M�<�:�A�M�Q�Z�s�����������y�s�f�Z���ݽ������������������鹝���������������ùϹعֹϹɹù����������ù¹��������ùϹܹ����������ܹϹù��O�L�C�O�Q�[�f�h�tāąĆā�~�t�j�h�[�O�O������ýþ������������������������������E\E[EPECE8E?ECEPE\EgEiErEuExEuEiE\E\E\E\������öíñù���������������������N�L�S�Z�`�g�s���������������������s�Z�N�����������������ĿɿѿԿ׿׿ѿĿ�����������������������������������������������������������������
����
�	����������D�D}D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� 4 A D > I  > e ! b J _ 2 Q | D ? D 9 � & ) 5 % d ? L e 9 f ^  9 > J m B L K M R / J I P T @ � t 9 / 1 F ( � O ` Z $ M 4 9 h H R . U V p Z ' 9 =    �  Z  �    �  �  W  Z  �  a  b  *  �  �  �  �  �  �  �  �    �  �  <    �  %  p  E  
  }  �  <  �  �  5  -  �    �  �  �  0  M  S    �  �    |  �  [  �  g  �  �  �  �  �  �  s  r  !  �    @  �  �  �    �  �  �  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  F/  d  s  }  �  �  �  �  s  i  s  �  �  �  �  e    �  y  �   �  i  X  F  5  #      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  ]  4  �  �  q  #  �  �  5  �  g  �  �  �  �  �  �  �  �    v  m  d  [  R  I  C  ?  :  6  1  �  �  %  B  W  c  `  O  2    �  �  �  o  $  �  &  S  P  �  <  0  &    4  !  �  �  �  �  �  ;  �  �  �  >  �    )  <  �  �  �  �  �  �  �  �  �  v  f  U  D  3  "         �   �  �  �  �  �  �  �  �  �  �  q  R  0    �  �  �  �  d  �  �  p  �  �  �  �  �  �            �  �  �  h  
  �  	  8  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  ,  u  q  n  i  d  _  X  Q  @  .    �  �  �  �  a  5    �  �  3    �  �  t  1  �  �  �  �  �  �  i  #  �  T    �  W   �  �  �  �  �  p  U  9    �  �  �  �  Y  &  �  S  �  T  �  >  8  (      �  �  �  �  �  �  k  O  2    �  �  �  k  6    �  �  �  �  �  �  �  y  m  a  T  E  6  '       �  �  �  �    E  l  �  �  �  �  �  w  g  A    �  d    �  �  �  ,  �  8  8  8  8  8  7  5  4  2  -  (  #      �  �  �  �  �  m  '  (  (  '  "            �  �  �  �  �  �  �  �  �  �  	]  	o  	m  	^  	@  	  �  �  P  �  �  7  �  ^  �  g    �  �  �  �  �  �  �  �  w  d  P  =  )    �  �  �  �  �  k  D     �  '  )  *  *  '  "        �  �  �  �  �  ^  6    �  �  ?  J  Q  W  Z  W  R  I  ?  3  $      �  �  �  �  �  �  w  ]  �  �  �  �  w  k  _  S  G  ;  -        �  �  �  �      �  �  �  �  �  �  �  �  �  �  
      �  �  }  G    �  �          �  �  �  �  �  �  �  �  c  0  �  �  I    �  �  C  C  B  @  <  7  /  '        �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  n  L  *    �  �  k    �  [  �  7  ]  �  �  �  �  }  x  r  m  h  c  ^  Z  V  R  O  K  G  C  ?  ;  B  <  3  %    �  �  �  �  �  i  D    �  �  �  j  /   �   _      �  �  �  �  y  X  :    �  �  �  �  t  6  �  y  �    �  �  �  �  �    {  w  s  o  e  U  D  4  $       �   �   �          �  �  �  �  �  �  �  �  �  �  m  O  5  	  �  �  �  �  �  �  �  �  �  �    k  T  :  !    �  �  �  �  �  i  �  �  z  Z  n  d  >    �  �  �  y  o  I    �  ~  %  �  �  
                �  �  �  �  h  G        �  q   �  �  �  �  �  �  �  �  �  �  �  �    	         (  0  8  ?  s  i  ^  T  I  =  1  %         �   �   �   �   �   �   �   �   �    /  (  &  )  +  $    �  �  �  �  y  P    �  W  �  .  !    �  �  �  �  �  �  �  �  k  N  /    �  �  �  �  �  h  O  �  �  �  �  �  �  �  r  ^  J  7  !    �  �  h     9  �  m  �  �    4  ?  =  3  $      �  �  �  b  &  �  ;  j  �   �  @  p  {  }  y  r  a  K  1    �  �  �  p  D    �  �  D  �  �  �          �  �  �  ~  T  '  �  �  �  X  "  �  X  �  	  �  �  �  �    S    �  y    �  B  �  N  �  '  a  �  �  z  u  q  l  h  c  _  Z  V  Q  K  B  :  1  )           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  m  �  �  �  �  �  {  s  l  d  \  X  V  T  R  Q  O  L  J  H  F  )  $              �  �  �  �  �  �  �  �  �  w  f  U      �  �  �  �  �  c  A     �  �  �  w  E    �  �  l  6  �  �  �  �    r  e  X  J  =  /  !      �  �  �  �  �  �  ,      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  6      �  �  �  �  �  �  �  �  �  �  �  �  �  s  c  R  B  2  �  �  �  �  �  �  x  ^  C  (    �  �  �  �  �  f  I     �  S  M  @  2  %    �  �  �  �  t  I    �  �  E  �  ,  c   �       �  �  �  �  �  �  �  �  �  �  o  \  H  (     �   �   �  �  �  �  {  o  a  P  >  *    �  �  �  �  �  z  [  9    �  O  B  6  *      �  �  �  �  �  �  S  "  �  �  �  M    �  <  4  '      �  �  �  �  �  l  G  $  �  �  �  J  &  9  G  �  �  �  �  �  m  W  =    �  �  �  �  \  ,  �  �  B  �  ^  �  �  �  �  �  t  �  �  �  h  E  "  �  �  �  v  ?  �  l  �  �  �  �  �  �  �  �  �  �  �  e  1  �  �  \  	  �  �  �  �  `  Y  R  L  E  >  7  1  +  %            �  �  �  �  �  �  �  �  �  |  `  B     �  �  �  a    �  �  @  �  �    �  J  C  @  N  U  M  <  ,  $  2    �  �  �  e    �  n    h        �  �  �  �  u  G    �  �  T     �  D  �  X  �  P    �  �  �  �  �  �  �  j  9    �  r    �    �     p   �  �  �  �  l  U  D  8  2  >  )    �  �  �  k  =    �  �  �  �      	  �  �  �  u  5  �  �  G  �  �  J  �  �  	  (  �  �  �  �  r  J    �  �  Y  $  �  �  F  �  �  y  o    �  V  2  )  !        �  �  �  �  �  _  4    �  �  �  t  I    �  �  �  �  �  �  |  c  I  ,    �  �  S    �  ~  /  �  n  f  W  I  <  ,      �  �  �  �  q  @    �  �  V    �  �  �  P  	  �  h  0  �  �  ]    �  ~  -  �  �  -  �  y    �